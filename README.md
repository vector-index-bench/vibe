<h1 align="center">VIBE</h1>
<div align="center">
Vector Index Benchmark for Embeddings (VIBE) is an extensible benchmark for approximate nearest neighbor search methods, or vector indexes, using modern embedding datasets.
</div>
<br/>

---

- Modern vector index benchmark with embedding datasets
- Includes datasets for both in-distribution and out-out-distribution settings
- Includes the most comprehensive collection of state-of-the-art vector search algorithms
- Support for quantized datasets in both 8-bit integer and binary precision
- Support for GPU algorithms
- Support for HPC environments with SLURM

### Results

The current VIBE results can be viewed on our website:

https://vector-index-bench.github.io

The website also features several other tools and visualizations to explore the results.

### Credits

The evaluation code in VIBE is based on the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks/) project.

### Datasets

| Name | Data Source | Model | Type | n | d | Distance |
|---|---|---|---|---|---|---|
| [agnews-mxbai-1024-euclidean](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/agnews-mxbai-1024-euclidean.hdf5) | AGNews | MXBAI | Text | 769,382 | 1024 | euclidean |
| [arxiv-nomic-768-normalized](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/arxiv-nomic-768-normalized.hdf5) | arXiv abstracts | Nomic Text | Text | 1,344,643 | 768 | any |
| [gooaq-distilroberta-768-normalized](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/gooaq-distilroberta-768-normalized.hdf5) | Google Q&A | DistilRoBERTa | Text | 1,475,024 | 768 | any |
| [imagenet-clip-512-normalized](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/imagenet-clip-512-normalized.hdf5) | ImageNet | CLiP | Image | 1,281,167 | 512 | any |
| [landmark-nomic-768-normalized](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/landmark-nomic-768-normalized.hdf5) | Landmark | Nomic Vision | Image | 760,757 | 768 | any |
| [yahoo-minilm-384-normalized](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/yahoo-minilm-384-normalized.hdf5) | Yahoo Answers | MiniLM | Text | 677,305 | 384 | any |
| [celeba-resnet-2048-cosine](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/celeba-resnet-2048-cosine.hdf5) | CelebA | ResNet | Image | 201,599 | 2048 | cosine |
| [ccnews-nomic-768-normalized](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/ccnews-nomic-768-normalized.hdf5) | CCNews | Nomic Text | Text | 495,328 | 768 | any |
| [codesearchnet-jina-768-cosine](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/codesearchnet-jina-768-cosine.hdf5) | CodeSearchNet | Jina | Code | 1,374,067 | 768 | cosine |
| [glove-200-cosine](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/glove-200-cosine.hdf5) | - | GloVe | Word | 1,192,514 | 200 | cosine |
| [landmark-dino-768-cosine](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/landmark-dino-768-cosine.hdf5) | Landmark | DINOv2 | Image | 760,757 | 768 | cosine |
| [simplewiki-openai-3072-normalized](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/simplewiki-openai-3072-normalized.hdf5) | Simple Wiki | OpenAI | Text | 260,372 | 3072 | any |

### Algorithms

| Method | Version |
|--------|---------|
| [ANNOY](https://github.com/spotify/annoy) | 1.17.3 |
| [FALCONN++](https://github.com/NinhPham/FalconnPP) | git+5fd3f17 |
| [FlatNav](https://github.com/BlaiseMuhirwa/flatnav) | 0.1.2 |
| [GLASS](https://github.com/zilliztech/pyglass) | 1.0.5 |
| [HNSW](https://github.com/nmslib/hnswlib) | 0.8.0 |
| [IVF (Faiss)](https://github.com/facebookresearch/faiss) | 1.11.0 |
| [IVF-PQ (Faiss)](https://github.com/facebookresearch/faiss) | 1.11.0 |
| [LVQ (SVS)](https://github.com/intel/ScalableVectorSearch) | 0.0.7 |
| [LeanVec (SVS)](https://github.com/intel/ScalableVectorSearch) | 0.0.7 |
| [LoRANN](https://github.com/ejaasaari/lorann) | 0.2 |
| [MLANN](https://github.com/ejaasaari/mlann) | git+40848e7 |
| [MRPT](https://github.com/vioshyvo/mrpt) | 2.0.1 |
| [NGT-ONNG](https://github.com/yahoojapan/NGT/) | git+83d5896 |
| [NGT-QG](https://github.com/yahoojapan/NGT/) | git+83d5896 |
| [NSG](https://github.com/facebookresearch/faiss) | 1.11.0 |
| [PUFFINN](https://github.com/puffinn/puffinn) | git+fd86b0d |
| [PyNNDescent](https://github.com/lmcinnes/pynndescent) | 0.5.13 |
| [RoarGraph](https://github.com/matchyc/RoarGraph) | git+f2b49b6 |
| [ScaNN](https://github.com/google-research/google-research/tree/master/scann) | 1.4.0 |
| [SymphonyQG](https://github.com/gouyt13/SymphonyQG) | git+32a0019 |
| [Vamana (DiskANN)](https://github.com/microsoft/DiskANN) | 0.7.0 |

## Getting started

### Requirements

- [Apptainer](https://apptainer.org/) (or [Singularity](https://sylabs.io/singularity/))
- Python 3.6+

Some algorithms may require that the CPU supports AVX-512 instructions. Most GPU algorithms assume that an NVIDIA GPU is available.

### Building library images

Building images can be done using `install.sh`. It can be used to either build images for all available libraries (`./install.sh`) or an image for a single library (e.g. `./install.sh --algorithm faiss`).

> [!TIP]
> `install.sh` takes an argument `--build-dir` that specifies the temporary build directory. For example, to speed up the build in a cluster environment, you can set the build directory to a location on an SSD while the project files are on a slower storage medium.

### Running benchmarks

The benchmarks for a single dataset can be run using `run.py`. For example:

```sh
python3 run.py --dataset agnews-mxbai-1024-euclidean
```

Common options for run.py:
- `--parallelism n`: Use `n` processes for benchmarking.
- `--count k`: Run the benchmarks using `k` nearest neighbors (default 100).
- `--gpu`: Run the benchmark in GPU mode.

### Creating datasets from scratch

The benchmark code downloads precomputed embedding datasets. However, the datasets can also be recreated from scratch, and it is also possible to create new datasets by modifying the [datasets.py](vibe/datasets.py) file.

Creating the datasets can be done using `create_dataset.sh`. It first requires that `dataset.sif` is built:
```sh
singularity build dataset.sif dataset.def
```

The VIBE_CACHE environment variable should be set to a cache directory with at least 200 GB of free space when creating image embeddings using the Landmark or ImageNet datasets. datasets can then be created using the `--dataset argument` (the `--nv` argument specifies that an available GPU can be used):
```sh
export VIBE_CACHE=$LOCAL_SCRATCH
./create_dataset "--bind $LOCAL_SCRATCH:$LOCAL_SCRATCH --nv" --dataset agnews-mxbai-1024-euclidean
```

## License

VIBE is available under the MIT License (see [LICENSE](LICENSE)). The [pyyaml](https://github.com/yaml/pyyaml) library is also distributed in the [vibe](vibe) folder under the MIT License.
