import os
import sys

VIBE_CACHE = os.environ.get("VIBE_CACHE", ".")
DATA_EXTRACT_DIR = f"{VIBE_CACHE}/data"
os.environ["HF_HOME"] = f"{VIBE_CACHE}/huggingface/cache"
os.environ["TORCH_HOME"] = f"{VIBE_CACHE}/torch/cache"

import random
import zipfile
import tarfile
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, Tuple

import h5py
import numpy
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from .runner import load_and_transform_dataset
from .util import download


def quantize_uint8(embeddings, starts, steps):
    """
    Quantizes embedding vectors to uint8 representation.

    Args:
        embeddings (numpy.ndarray): Input embedding vectors to be quantized.
        starts (numpy.ndarray): Starting values for quantization range.
        steps (numpy.ndarray): Step sizes for quantization.

    Returns:
        numpy.ndarray: Quantized embeddings as uint8 values.
    """
    import numpy

    return ((embeddings - starts) / steps).astype(numpy.uint8)


def quantize_binary(embeddings):
    """
    Converts embedding vectors to binary representation by thresholding at zero.

    Args:
        embeddings (numpy.ndarray): Input embedding vectors to be binarized.

    Returns:
        numpy.ndarray: Binary representation of embeddings packed into bytes (uint8).
    """
    import numpy

    return numpy.packbits(embeddings > 0).reshape(embeddings.shape[0], -1)


def normalize_embeddings(embeddings):
    """
    Normalize each row of a matrix to have unit length (L2 norm).

    Args:
        embeddings (numpy.ndarray): Input embedding vectors to be reduced.

    Returns:
        numpy.ndarray: Normalized embedding vectors.
    """
    import numpy

    embeddings[numpy.linalg.norm(embeddings, axis=1) == 0] = 1.0 / numpy.sqrt(embeddings.shape[1])
    embeddings /= numpy.linalg.norm(embeddings, axis=1)[:, numpy.newaxis]
    return numpy.ascontiguousarray(embeddings, dtype=numpy.float32)


def reduce_embeddings(embeddings, dim):
    """
    Reduces embedding dimensionality.

    Args:
        embeddings (numpy.ndarray): Input embedding vectors to be reduced.
        dim (int): Target dimensionality for the reduced embeddings.

    Returns:
        numpy.ndarray: Reduced embedding vectors.
    """
    return normalize_embeddings(normalize_embeddings(embeddings)[:, :dim])


def write_output(
    fn: str,
    train: numpy.ndarray,
    test: numpy.ndarray,
    learn: numpy.ndarray = None,
    distance: str = "cosine",
    point_type: str = "float",
    count: int = 100,
) -> None:
    """
    Writes the provided training and testing data to an HDF5 file. It also computes
    and stores the nearest neighbors and their distances for the test set using a
    brute-force approach.

    Args:
        filename (str): The name of the HDF5 file to which data should be written.
        train (numpy.ndarray): The training data.
        test (numpy.ndarray): The testing data.
        learn (numpy.ndarray): Optional learning data from the query distribution.
        distance (str): The distance metric to use for computing nearest neighbors.
            Defaults to "cosine".
        point_type (str, optional): The type of the data points. Defaults to "float".
        count (int, optional): The number of nearest neighbors to compute for
            each point in the test set. Defaults to 100.
    """
    from vibe.algorithms.faiss.module import FaissFlat

    dimension = train.shape[1]
    print(f"train size: {train.shape[0]} * {train.shape[1]}")
    print(f"test size: {test.shape[0]} * {test.shape[1]}")

    if point_type == "uint8":
        if distance != "euclidean":
            raise ValueError("Only Euclidean distance is supported with uint8 precision")

        print("Quantizing to uint8 precision")
        ranges = numpy.vstack((numpy.min(train, axis=0), numpy.max(train, axis=0)))
        starts = ranges[0, :]
        steps = (ranges[1, :] - ranges[0, :]) / 255
        train = quantize_uint8(train, starts, steps)
        test = quantize_uint8(test, starts, steps)
        if learn is not None:
            learn = quantize_uint8(learn, starts, steps)

    elif point_type == "binary":
        if distance != "hamming":
            raise ValueError("Only Hamming distance is supported with binary precision")

        print("Quantizing to binary precision")
        train = quantize_binary(train)
        test = quantize_binary(test)
        if learn is not None:
            learn = quantize_binary(learn)

    if distance == "normalized":
        train = normalize_embeddings(train)
        test = normalize_embeddings(test)
        if learn is not None:
            learn = normalize_embeddings(learn)

    with h5py.File(fn, "w") as f:
        f.attrs["distance"] = distance
        f.attrs["dimension"] = dimension
        f.attrs["point_type"] = point_type

        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)

        neighbors_ds = f.create_dataset("neighbors", (len(test), count), dtype=int)
        distances_ds = f.create_dataset("distances", (len(test), count), dtype=float)
        avg_dists_ds = f.create_dataset("avg_distances", len(test), dtype=float)

        # Fit the brute-force k-NN model
        bf = FaissFlat(distance)
        bf.fit(train)

        print("Computing true k-nn for test...")
        for i, x in enumerate(test):
            # Query the model and sort results by distance
            res, avg_dist = bf.query_with_distances(x, count, return_avg_dist=True)
            res = list(res)
            res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            neighbors_ds[i] = [idx for idx, _ in res]
            distances_ds[i] = [dist for _, dist in res]
            avg_dists_ds[i] = avg_dist

        if learn is not None and learn.dtype == numpy.float32:
            print(f"learn size: {learn.shape[0]} * {learn.shape[1]}")
            f.create_dataset("learn", data=learn)

            print("Computing true k-nn for learn...")
            if torch.cuda.is_available():
                from vibe.algorithms.cuvs.module import cuVSBruteForce

                cuda_bf = cuVSBruteForce(distance)
                cuda_bf.fit(train)
                cuda_bf.batch_query(learn, count)
                learn_neighbors = cuda_bf.get_batch_results()
            else:
                print("Warning: CUDA not available, this might be slow...")
                bf.batch_query(learn, count)
                learn_neighbors = bf.get_batch_results().astype(numpy.uint32)

            f.create_dataset("learn_neighbors", data=learn_neighbors)


def train_test_split(
    X: numpy.ndarray, test_size: int = 1000, dimension: int = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Splits the provided dataset into a training set and a testing set.

    Args:
        X (numpy.ndarray): The dataset to split.
        test_size (int, optional): The number of samples to include in the test set.
            Defaults to 1000.
        dimension (int, optional): The dimensionality of the data. If not provided,
            it will be inferred from the second dimension of X. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the training set and the testing set.
    """
    from sklearn.model_selection import train_test_split as sklearn_train_test_split

    dimension = dimension if dimension is not None else X.shape[1]
    print(f"Splitting {X.shape[0]}*{dimension} into train/test")
    return sklearn_train_test_split(X, test_size=test_size, random_state=1)


def random_sample(X: numpy.ndarray, size: int = 1000, seed: int = 1) -> numpy.ndarray:
    """
    Randomly samples a subset of data points from the provided dataset.

    Args:
        X (numpy.ndarray): The dataset to sample from.
        size (int, optional): The number of samples to draw from the dataset.
            Defaults to 1000.
        seed (int, optional): Random seed for reproducibility.
            Defaults to 1.

    Returns:
        numpy.ndarray: A randomly sampled subset of the input dataset.
    """
    numpy.random.seed(seed)
    random_idx = numpy.random.choice(numpy.arange(len(X)), size=size, replace=False)
    return X[random_idx]


def extract_archive(archive_file, extract_dir, remove=False):
    print("Extracting %s..." % archive_file)

    if archive_file.endswith(".tar"):
        with tarfile.open(archive_file, "r") as tar:
            tar.extractall(path=extract_dir)
    elif archive_file.endswith(".zip"):
        with zipfile.ZipFile(archive_file, "r") as zip_ref:
            zip_ref.extractall(path=extract_dir)
    else:
        return

    if remove:
        os.remove(archive_file)


def extract_dir(path, target_dir, remove=False):
    archive_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith((".zip", ".tar")):
                archive_files.append(os.path.join(root, file))

    extract_fn = partial(extract_archive, extract_dir=target_dir, remove=remove)

    max_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(extract_fn, archive_files)


def get_all_image_paths(path):
    image_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.split(".")[-1].lower() in ["jpg", "jpeg", "png"]:
                image_paths.append(os.path.join(root, file))
    return image_paths


class ArchivedImageDataset(Dataset):
    def __init__(self, archive, target_dir, transform):
        os.makedirs(target_dir, exist_ok=True)

        if not os.listdir(target_dir):
            if os.path.isdir(archive):
                extract_dir(archive, target_dir, remove=False)
            else:
                extract_archive(archive, target_dir)

            extract_dir(target_dir, target_dir, remove=True)

        self.image_paths = get_all_image_paths(target_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)


class HuggingFaceDataset(Dataset):
    def __init__(self, name, attribute, subset=None, deduplicate=True):
        from datasets import load_dataset

        if subset is None:
            ds = load_dataset(name, split="train")
        else:
            ds = load_dataset(name, subset, split="train")

        if isinstance(attribute, list):
            documents = [" ".join(ds[a][i] for a in attribute) for i in range(len(ds[attribute[0]]))]
        else:
            documents = ds[attribute]

        if deduplicate:
            documents = list(set(documents))

        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]


class COCOAnnotationsDataset(Dataset):
    def __init__(self, archive_file, extract_dir, split="train", sample=None, deduplicate=True):
        import json
        import collections

        extract_archive(archive_file, extract_dir, remove=False)
        data = json.load(open(f"{extract_dir}/annotations/captions_{split}2017.json", "r"))

        annotations = collections.defaultdict(list)
        for annotation in data["annotations"]:
            annotations[annotation["image_id"]].append(annotation["caption"])

        documents = [captions[0] for _, captions in annotations.items()]

        if deduplicate:
            documents = list(set(documents))

        if sample is not None:
            random.seed(42)
            self.documents = random.sample(documents, sample)
        else:
            self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]


def clip_image_embedding():
    from transformers import CLIPProcessor, CLIPModel

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)

    def transform(image):
        return processor(images=image, return_tensors="pt", padding=True, device=device)["pixel_values"].squeeze(0)

    def f(images):
        with torch.no_grad():
            return model.get_image_features(pixel_values=images).cpu().numpy()

    return f, transform


def clip_text_embedding():
    from transformers import CLIPProcessor, CLIPModel

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)

    def f(texts):
        with torch.no_grad():
            input_ids = processor(text=texts, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            return model.get_text_features(input_ids=input_ids.to(device)).cpu().numpy()

    return f


def perception_image_embedding():
    sys.path.append("/perception_models")

    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)
    model = model.to(device)
    preprocess = transforms.get_image_transform(model.image_size)

    def transform(image):
        return preprocess(image).to(device)

    def f(images):
        with torch.no_grad():
            image_features, _, _ = model(images)
            return image_features.cpu().numpy()

    return f, transform


def perception_text_embedding():
    sys.path.append("/perception_models")

    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)
    model = model.to(device)
    tokenizer = transforms.get_text_tokenizer(model.context_length)

    def f(sentences):
        with torch.no_grad():
            _, text_features, _ = model(text=tokenizer(sentences).to(device))
            return text_features.cpu().numpy()

    return f


def flava_image_embedding():
    import torch.nn.functional as F
    from transformers import FlavaModel, FlavaProcessor

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = FlavaModel.from_pretrained("facebook/flava-full").to(device).eval()
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")

    def transform(image):
        return processor(images=image, return_tensors="pt").to(device)["pixel_values"].squeeze(0)

    def f(images):
        with torch.no_grad():
            out = model(pixel_values=images, skip_multimodal_encoder=True)
            tok_emb = out.image_embeddings
            cls_emb = tok_emb[:, 0, :]
            img_vec = F.normalize(model.image_projection(cls_emb), dim=-1)
            return img_vec.cpu().numpy()

    return f, transform


def flava_text_embedding():
    import torch.nn.functional as F
    from transformers import FlavaModel, FlavaProcessor

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = FlavaModel.from_pretrained("facebook/flava-full").to(device).eval()
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")

    def f(sentences):
        with torch.no_grad():
            txt_inputs = processor(text=sentences, padding=True, return_tensors="pt").to(device)
            out = model(
                input_ids=txt_inputs["input_ids"],
                attention_mask=txt_inputs["attention_mask"],
                skip_multimodal_encoder=True,
            )
            tok_emb = out.text_embeddings
            txt_vec = F.normalize(tok_emb[:, 0, :], dim=-1)
            return txt_vec.cpu().numpy()

    return f


def align_image_embedding():
    import torch.nn.functional as F
    from transformers import AlignProcessor, AlignModel

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = AlignModel.from_pretrained("kakaobrain/align-base")
    model = model.to(device)
    processor = AlignProcessor.from_pretrained("kakaobrain/align-base")

    def transform(image):
        return processor(images=image, return_tensors="pt").to(device)["pixel_values"].squeeze(0)

    def f(images):
        with torch.no_grad():
            img_vec = model.get_image_features(pixel_values=images)
            return F.normalize(img_vec, dim=-1).cpu().numpy()

    return f, transform


def align_text_embedding():
    import torch.nn.functional as F
    from transformers import AlignProcessor, AlignModel

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = AlignModel.from_pretrained("kakaobrain/align-base")
    model = model.to(device)
    processor = AlignProcessor.from_pretrained("kakaobrain/align-base")

    def f(sentences):
        with torch.no_grad():
            txt_inputs = processor(text=sentences, padding=True, return_tensors="pt").to(device)
            txt_vecs = model.get_text_features(**txt_inputs)
            txt_vecs = F.normalize(txt_vecs, dim=-1)
            return txt_vecs.cpu().numpy()

    return f


def nomic_vision_embedding():
    import torch.nn.functional as F
    from transformers import AutoModel, AutoImageProcessor

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", use_fast=True)
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
    model = model.to(device)

    def transform(image):
        return processor(images=image, return_tensors="pt", device=device)["pixel_values"].squeeze(0)

    def f(images):
        with torch.no_grad():
            img_emb = model(pixel_values=images).last_hidden_state
            embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
            return embeddings.cpu().numpy()

    return f, transform


def nomic_embed(prefix="search_document"):
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    def f(sentences):
        new_sentences = [f"{prefix}: {sentence}" for sentence in sentences]

        embeddings = model.encode(new_sentences, convert_to_tensor=True)
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    return f


def dino_embedding():
    import timm

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = timm.create_model(
        "vit_base_patch16_224.dino",
        pretrained=True,
        num_classes=0,
    )
    model = model.to(device)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    def f(images):
        with torch.no_grad():
            return model(images).cpu().numpy()

    return f, transform


def swin_embedding():
    import timm

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = timm.create_model(
        "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        pretrained=True,
        num_classes=0,
    )
    model = model.to(device)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    def f(images):
        with torch.no_grad():
            return model(images).cpu().numpy()

    return f, transform


def resnet_embedding():
    from torchvision import models

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()

    model = model.to(device)
    model = model.eval()

    transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()

    def f(images):
        with torch.no_grad():
            return model(images).cpu().numpy()

    return f, transform


def mobilenet_embedding():
    from torchvision import models

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()

    model = model.to(device)
    model = model.eval()

    transform = models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()

    def f(images):
        with torch.no_grad():
            return model(images).cpu().numpy()

    return f, transform


def litellm_embed(model, max_tokens=8192):
    import litellm

    def f(sentences):
        tokenized = [litellm.encode(model=model, text=s)[:max_tokens] for s in sentences]
        res = litellm.embedding(model=model, input=tokenized)
        embeddings = [x["embedding"] for x in res.data]
        return numpy.array(embeddings, dtype=numpy.float32)

    return f


def minilm_embed(doc_type="corpus"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def f(sentences):
        return model.encode(sentences)

    return f


def mpnet_embed(doc_type="corpus"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def f(sentences):
        return model.encode(sentences)

    return f


def distilroberta_embed(doc_type="corpus"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

    def f(sentences):
        return model.encode(sentences)

    return f


def mxbai_embed(doc_type="corpus"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    def f(sentences):
        if doc_type == "query":
            return model.encode(sentences, prompt_name="query")
        return model.encode(sentences)

    return f


def snowflake_embed(model="arctic-embed-m-v2.0", doc_type="corpus"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(f"Snowflake/snowflake-{model}", trust_remote_code=True)

    def f(sentences):
        if doc_type == "query":
            return model.encode(sentences, prompt_name="query")
        return model.encode(sentences)

    return f


def codet5p_embed():
    from transformers import AutoModel, AutoTokenizer

    checkpoint = "Salesforce/codet5p-110m-embedding"
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    model = model.to(device)

    def f(codes):
        inputs = tokenizer(codes, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            return model(**inputs).cpu().numpy()

    return f


def nomic_code_embed():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True)
    model.max_seq_length = 256

    def f(codes):
        return model.encode(codes)

    return f


def qodo_embed():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Qodo/Qodo-Embed-1-1.5B")
    model.max_seq_length = 256

    def f(codes):
        return model.encode(codes)

    return f


def jina_code_embed():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True)
    model.max_seq_length = 256

    def f(codes):
        return model.encode(codes)

    return f


def jina_embed(task="text-matching"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

    def f(sentences):
        return model.encode(sentences, task=task, prompt_name=task)

    return f


def glove(out_fn: str, d: int) -> None:
    import zipfile

    url = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    fn = os.path.join("data", "glove.twitter.27B.zip")
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print("preparing %s" % out_fn)
        z_fn = "glove.twitter.27B.%dd.txt" % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        X = numpy.array(X).astype(numpy.float32)
        X_train, X_test = train_test_split(X)
        write_output(out_fn, numpy.array(X_train), numpy.array(X_test), distance="cosine")


def embedding_dataset(
    out_fn,
    corpus_dataloader,
    query_dataloader,
    embedding,
    query_embedding=None,
    learn_dataloader=None,
    metric="cosine",
):
    if query_embedding is None:
        query_embedding = embedding

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cpu"

    corpus_embeddings = []
    for batch in tqdm.tqdm(corpus_dataloader):
        if torch.is_tensor(batch):
            batch = batch.to(device)
        corpus_embeddings.append(embedding(batch))
    corpus_embeddings = numpy.vstack(corpus_embeddings)

    query_embeddings = []
    for batch in tqdm.tqdm(query_dataloader):
        if torch.is_tensor(batch):
            batch = batch.to(device)
        query_embeddings.append(query_embedding(batch))
    query_embeddings = numpy.vstack(query_embeddings)

    learn_embeddings = None
    if learn_dataloader is not None:
        learn_embeddings = []
        for batch in tqdm.tqdm(learn_dataloader):
            if torch.is_tensor(batch):
                batch = batch.to(device)
            learn_embeddings.append(query_embedding(batch))
        learn_embeddings = numpy.vstack(learn_embeddings)

    write_output(out_fn, corpus_embeddings, query_embeddings, learn_embeddings, distance=metric)


def text_embedding_dataset(
    out_fn,
    dataset_name,
    attribute,
    query_attribute,
    embedding,
    query_embedding=None,
    subset=None,
    ood=False,
    metric="cosine",
):
    test_size = 1000
    batch_size = 256

    dataset = HuggingFaceDataset(dataset_name, attribute, subset)
    generator = torch.Generator().manual_seed(42)

    if query_attribute is None or not ood:
        corpus, queries = random_split(dataset, [len(dataset) - test_size, test_size], generator=generator)
        corpus_dataloader = DataLoader(corpus, batch_size=batch_size, shuffle=False, num_workers=0)
        query_dataloader = DataLoader(queries, batch_size=batch_size, shuffle=False, num_workers=0)
        learn_dataloader = None
    else:
        query_dataset = HuggingFaceDataset(dataset_name, query_attribute, subset)
        learn, queries = random_split(query_dataset, [len(query_dataset) - test_size, test_size], generator=generator)
        corpus_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        query_dataloader = DataLoader(queries, batch_size=batch_size, shuffle=False, num_workers=0)
        learn_dataloader = DataLoader(learn, batch_size=batch_size, shuffle=False, num_workers=0)

    embedding_dataset(
        out_fn, corpus_dataloader, query_dataloader, embedding, query_embedding, learn_dataloader, metric=metric
    )


def celeba_loader(transform):
    data_dir = "./data/celebA/"
    fn = "img_align_celeba.zip"

    os.makedirs(data_dir, exist_ok=True)
    download("https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/img_align_celeba.zip", data_dir + fn)

    return ArchivedImageDataset(data_dir + fn, DATA_EXTRACT_DIR + "/celeba", transform)


def landmark_loader(transform):
    data_dir = "./data/landmark/"

    os.makedirs(data_dir, exist_ok=True)
    for i in range(100):
        fn = "images_%03d.tar" % i
        download("https://s3.amazonaws.com/google-landmark/index/" + fn, data_dir + fn)

    return ArchivedImageDataset(data_dir, DATA_EXTRACT_DIR + "/landmark", transform)


def imagenet_loader(split, transform):
    data_dir = "./data/imagenet/"

    if split == "train":
        fn = "ILSVRC2012_img_train.tar"
    else:
        fn = "ILSVRC2012_img_test_v10102019.tar"

    if not os.path.isfile(data_dir + fn):
        raise Exception(f"You must download {fn} manually and place it in {data_dir + fn}")

    return ArchivedImageDataset(data_dir + fn, DATA_EXTRACT_DIR + "/imagenet/" + split, transform)


def coco_image_loader(transform):
    data_dir = "./data/coco/"

    os.makedirs(data_dir, exist_ok=True)

    download("http://images.cocodataset.org/zips/train2017.zip", data_dir + "train2017.zip")
    download("http://images.cocodataset.org/zips/test2017.zip", data_dir + "test2017.zip")
    download("http://images.cocodataset.org/zips/unlabeled2017.zip", data_dir + "unlabeled2017.zip")

    return ArchivedImageDataset(data_dir, DATA_EXTRACT_DIR + "/coco/", transform)


def coco_text_loader(split, sample=None):
    data_dir = "./data/coco/"
    fn = "annotations_trainval2017.zip"

    os.makedirs(data_dir, exist_ok=True)

    download("http://images.cocodataset.org/annotations/" + fn, data_dir + fn)

    return COCOAnnotationsDataset(data_dir + fn, DATA_EXTRACT_DIR + "/coco/", split, sample)


def image_embedding_dataset(out_fn, dataset, embedding, query_dataset=None, query_embedding=None, metric="cosine"):
    test_size = 1000
    batch_size = 128

    generator = torch.Generator().manual_seed(42)
    if query_dataset is None:
        corpus, queries = random_split(dataset, [len(dataset) - test_size, test_size], generator=generator)
        corpus_dataloader = DataLoader(corpus, batch_size=batch_size, shuffle=False, num_workers=0)
        query_dataloader = DataLoader(queries, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        corpus_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        sampler = RandomSampler(query_dataset, replacement=False, num_samples=test_size, generator=generator)
        query_dataloader = DataLoader(
            query_dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=sampler
        )

    embedding_dataset(out_fn, corpus_dataloader, query_dataloader, embedding, query_embedding, metric=metric)


def imagenet(out_fn, embedding, metric="cosine"):
    embedding_func, transform = embedding()
    image_embedding_dataset(
        out_fn,
        imagenet_loader("train", transform),
        embedding_func,
        imagenet_loader("test", transform),
        embedding_func,
        metric,
    )


def celeba(out_fn, embedding, metric="cosine"):
    embedding_func, transform = embedding()
    image_embedding_dataset(out_fn, celeba_loader(transform), embedding_func, None, embedding_func, metric)


def landmark(out_fn, embedding, metric="cosine"):
    embedding_func, transform = embedding()
    image_embedding_dataset(out_fn, landmark_loader(transform), embedding_func, None, embedding_func, metric)


def simplewiki(out_fn, embedding, query_embedding=None, metric="normalized"):
    text_embedding_dataset(
        out_fn, "ejaasaari/simplewiki-2025-04-01", "text", None, embedding, query_embedding, metric=metric
    )


def arxiv(out_fn, embedding, query_embedding=None, metric="normalized"):
    text_embedding_dataset(
        out_fn, "ejaasaari/arxiv-abstracts-2025-04-13", "text", None, embedding, query_embedding, metric=metric
    )


def ccnews(out_fn, embedding, query_embedding=None, metric="normalized"):
    text_embedding_dataset(
        out_fn, "sentence-transformers/ccnews", "article", None, embedding, query_embedding, metric=metric
    )


def gooaq(out_fn, embedding, query_embedding=None, metric="normalized"):
    text_embedding_dataset(
        out_fn, "sentence-transformers/gooaq", "answer", "question", embedding, query_embedding, metric=metric
    )


def yahoo_answers(out_fn, embedding, query_embedding=None, metric="normalized"):
    text_embedding_dataset(
        out_fn,
        "sentence-transformers/yahoo-answers",
        "answer",
        "question",
        embedding,
        query_embedding,
        subset="question-answer-pair",
        metric=metric,
    )


def agnews(out_fn, embedding, query_embedding=None, metric="normalized"):
    text_embedding_dataset(
        out_fn, "sentence-transformers/agnews", "description", None, embedding, query_embedding, metric=metric
    )


def codesearchnet(out_fn, embedding, query_embedding=None, metric="normalized"):
    text_embedding_dataset(
        out_fn, "sentence-transformers/codesearchnet", "code", None, embedding, query_embedding, metric=metric
    )


def _read_fbin(filename: str, start_row: int = 0, count_rows=None):
    import numpy as np

    dtype = np.float32
    scalar_size = 4

    with open(filename, "rb") as f:
        rows, cols = np.fromfile(f, count=2, dtype=np.int32)
        rows = (rows - start_row) if count_rows is None else count_rows
        arr = np.fromfile(f, count=rows * cols, dtype=dtype, offset=start_row * scalar_size * cols)

    return arr.reshape(rows, cols)


def yandex(out_fn, metric="cosine"):
    nb = 1_000_000
    file_size = 8 + 200 * nb * numpy.dtype("float32").itemsize

    train_out_file = "data/yandex.fbin"
    learn_out_file = "data/yandex_learn.fbin"
    query_out_file = "data/yandex_query.fbin"
    download(
        "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin",
        train_out_file,
        max_size=file_size,
    )
    download(
        "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin",
        learn_out_file,
        max_size=file_size,
    )

    for out_file in [train_out_file, learn_out_file]:
        header = numpy.memmap(out_file, shape=2, dtype="uint32", mode="r+")
        assert header[1] == 200
        header[0] = nb

    download(
        "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin",
        query_out_file,
    )

    X_train = _read_fbin(train_out_file)
    X_learn = _read_fbin(learn_out_file)
    X_test = random_sample(_read_fbin(query_out_file), size=1000)

    write_output(out_fn, X_train, X_test, X_learn, distance=metric)


def llama(out_fn, layer, head, metric="ip"):
    download(
        f"https://huggingface.co/datasets/ejaasaari/llama_heads/resolve/main/K_{layer}_{head}.npy",
        f"data/llama_K_{layer}_{head}.npy",
    )
    download(
        f"https://huggingface.co/datasets/ejaasaari/llama_heads/resolve/main/Q_{layer}_{head}.npy",
        f"data/llama_Q_{layer}_{head}.npy",
    )

    K = numpy.load(f"data/llama_K_{layer}_{head}.npy")[1:-2047]
    Q = numpy.load(f"data/llama_Q_{layer}_{head}.npy")[1:-2047]
    X_learn, X_test = train_test_split(Q, test_size=1000)

    write_output(out_fn, K, X_test, X_learn, distance=metric)


def yi(out_fn, layer, head, metric="ip"):
    download(
        f"https://huggingface.co/datasets/ejaasaari/yi_heads/resolve/main/K_{layer}_{head}.npy",
        f"data/yi_K_{layer}_{head}.npy",
    )
    download(
        f"https://huggingface.co/datasets/ejaasaari/yi_heads/resolve/main/Q_{layer}_{head}.npy",
        f"data/yi_Q_{layer}_{head}.npy",
    )

    K = numpy.load(f"data/yi_K_{layer}_{head}.npy")[1:-2047]
    Q = numpy.load(f"data/yi_Q_{layer}_{head}.npy")[1:-2047]
    X_learn, X_test = train_test_split(Q, test_size=1000)

    write_output(out_fn, K, X_test, X_learn, distance=metric)


def coco(out_fn, image_embedding, text_embedding, metric="cosine"):
    batch_size = 256

    image_embedding_func, image_transform = image_embedding()
    corpus_dataset = coco_image_loader(image_transform)
    corpus_dataloader = DataLoader(corpus_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    learn_dataset = coco_text_loader("train")
    learn_dataloader = DataLoader(learn_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    query_dataset = coco_text_loader("val", sample=1000)
    query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embedding_dataset(
        out_fn,
        corpus_dataloader,
        query_dataloader,
        image_embedding_func,
        text_embedding,
        learn_dataloader,
        metric,
    )


def imagenet_captions(out_fn, image_embedding, text_embedding, metric="normalized"):
    batch_size = 128
    test_size = 1000

    image_embedding_func, image_transform = image_embedding()
    corpus_dataset = imagenet_loader("train", image_transform)
    corpus_dataloader = DataLoader(corpus_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    query_dataset = HuggingFaceDataset("ejaasaari/imagenet-captions", "text")

    generator = torch.Generator().manual_seed(42)
    learn, queries = random_split(query_dataset, [len(query_dataset) - test_size, test_size], generator=generator)
    query_dataloader = DataLoader(queries, batch_size=batch_size, shuffle=False, num_workers=0)
    learn_dataloader = DataLoader(learn, batch_size=batch_size, shuffle=False, num_workers=0)

    embedding_dataset(
        out_fn,
        corpus_dataloader,
        query_dataloader,
        image_embedding_func,
        text_embedding,
        learn_dataloader,
        metric,
    )


def laion(out_fn, metric="cosine"):
    download(
        "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/images/img_emb_0.npy",
        "img_emb_0.npy",
    )
    download(
        "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/texts/text_emb_0.npy",
        "text_emb_0.npy",
    )

    X = numpy.load("img_emb_0.npy").astype(numpy.float32)
    Q = numpy.load("text_emb_0.npy").astype(numpy.float32)
    X_learn, X_test = train_test_split(Q, test_size=1000)

    write_output(out_fn, X, X_test, X_learn, distance=metric)


def binary_dataset(out_fn, original_dataset):
    try:
        train, test, _ = load_and_transform_dataset(original_dataset)
    except:
        raise ValueError(f"failed to load {original_dataset} -- make sure the original dataset exists first")

    write_output(out_fn, train, test, distance="hamming", point_type="binary")


def uint8_dataset(out_fn, original_dataset):
    try:
        train, test, _ = load_and_transform_dataset(original_dataset)
    except:
        raise ValueError(f"failed to load {original_dataset} -- make sure the original dataset exists first")

    write_output(out_fn, train, test, distance="euclidean", point_type="uint8")


def matryoshka_dataset(out_fn, original_dataset, new_dim):
    try:
        train, test, _ = load_and_transform_dataset(original_dataset)
    except:
        raise ValueError(f"failed to load {original_dataset} -- make sure the original dataset exists first")

    train_reduced = reduce_embeddings(train, new_dim)
    test_reduced = reduce_embeddings(test, new_dim)

    write_output(out_fn, train_reduced, test_reduced, distance="normalized")


def ood_to_id_dataset(out_fn, original_dataset):
    try:
        train, _, distance = load_and_transform_dataset(original_dataset)
    except:
        raise ValueError(f"failed to load {original_dataset} -- make sure the original dataset exists first")

    X_train, X_test = train_test_split(train, test_size=1000)

    write_output(out_fn, X_train, X_test, X_train, distance=distance)


DATASETS: Dict[str, Callable[[str], None]] = {
    "agnews-mxbai-1024-euclidean": lambda out_fn: agnews(out_fn, mxbai_embed(), metric="euclidean"),
    "agnews-mxbai-1024-euclidean-uint8": lambda out_fn: uint8_dataset(out_fn, "agnews-mxbai-1024-euclidean"),
    "agnews-mxbai-1024-hamming-binary": lambda out_fn: binary_dataset(out_fn, "agnews-mxbai-1024-euclidean"),
    "arxiv-nomic-768-normalized": lambda out_fn: arxiv(
        out_fn, nomic_embed("clustering"), nomic_embed("clustering"), metric="normalized"
    ),
    "arxiv-nomic-768-euclidean-uint8": lambda out_fn: uint8_dataset(out_fn, "arxiv-nomic-768-normalized"),
    "arxiv-nomic-768-hamming-binary": lambda out_fn: binary_dataset(out_fn, "arxiv-nomic-768-normalized"),
    "ccnews-nomic-768-euclidean-uint8": lambda out_fn: uint8_dataset(out_fn, "ccnews-nomic-768-normalized"),
    "ccnews-nomic-768-hamming-binary": lambda out_fn: binary_dataset(out_fn, "ccnews-nomic-768-normalized"),
    "ccnews-nomic-768-normalized": lambda out_fn: ccnews(out_fn, nomic_embed()),
    "celeba-resnet-2048-cosine": lambda out_fn: celeba(out_fn, resnet_embedding),
    "coco-nomic-768-normalized": lambda out_fn: coco(
        out_fn, nomic_vision_embedding, nomic_embed("query"), metric="normalized"
    ),
    "codesearchnet-jina-768-cosine": lambda out_fn: codesearchnet(out_fn, jina_code_embed(), metric="cosine"),
    "glove-200-cosine": lambda out_fn: glove(out_fn, 200),
    "gooaq-distilroberta-768-normalized": lambda out_fn: gooaq(out_fn, distilroberta_embed()),
    "imagenet-align-640-normalized": lambda out_fn: imagenet_captions(
        out_fn, align_image_embedding, align_text_embedding(), metric="normalized"
    ),
    "imagenet-clip-512-normalized": lambda out_fn: imagenet(out_fn, clip_image_embedding, metric="normalized"),
    "laion-clip-512-normalized": lambda out_fn: laion(out_fn, "normalized"),
    "landmark-dino-768-cosine": lambda out_fn: landmark(out_fn, dino_embedding),
    "landmark-nomic-768-euclidean-uint8": lambda out_fn: uint8_dataset(out_fn, "landmark-nomic-768-normalized"),
    "landmark-nomic-768-hamming-binary": lambda out_fn: binary_dataset(out_fn, "landmark-nomic-768-normalized"),
    "landmark-nomic-768-normalized": lambda out_fn: landmark(out_fn, nomic_vision_embedding, metric="normalized"),
    "llama-128-ip": lambda out_fn: llama(out_fn, 12, 15),
    "simplewiki-openai-3072-euclidean-uint8": lambda out_fn: uint8_dataset(out_fn, "simplewiki-openai-3072-normalized"),
    "simplewiki-openai-3072-hamming-binary": lambda out_fn: binary_dataset(out_fn, "simplewiki-openai-3072-normalized"),
    "simplewiki-openai-3072-normalized": lambda out_fn: simplewiki(out_fn, litellm_embed("text-embedding-3-large")),
    "yahoo-minilm-384-normalized": lambda out_fn: yahoo_answers(out_fn, minilm_embed()),
    "yandex-200-cosine": lambda out_fn: yandex(out_fn, "cosine"),
    "yi-128-ip": lambda out_fn: yi(out_fn, 31, 13),
}
