import numpy as np
import torch
import os
import tqdm
import random
import faiss


def sample_itemID4kmeans(n_item):
    # Simple alternative: < 100k: 100%, < 1M: 15%, < 10M: 7%, < 100M: 3%, > 100M: 1%
    # Keep in mind that, say, 15% still means at least 100k.
    # So the formula is max(100% * min(total, 100k), 15% * min(total, 1M), ...)
    # Then we subsample the vectors to 100 * num_partitions

    typical_doclen = 120  # let's keep sampling independent of the actual doc_maxlen
    n_sample_pid = 8 * np.sqrt(typical_doclen * n_item)
    # n_sample_pid = np.sqrt(typical_doclen * n_item) / 2
    # sampled_pids = int(2 ** np.floor(np.log2(1 + sampled_pids)))
    n_sample_pid = min(1 + int(n_sample_pid), n_item)

    random.seed(12345)
    sample_pid_l = random.sample(range(n_item), n_sample_pid)

    return sample_pid_l


def get_sample_vecs_l(sample_itemID_l: list, DEFAULT_CHUNKSIZE: int, train: np.ndarray, train_counts: np.ndarray):
    sample_itemID_l = np.sort(sample_itemID_l)
    item_chunkID_l = [_ // DEFAULT_CHUNKSIZE for _ in sample_itemID_l]
    item_chunk_offset_l = [_ % DEFAULT_CHUNKSIZE for _ in sample_itemID_l]
    chunkID2offset_m = {}
    for chunkID, chunk_offset in zip(item_chunkID_l, item_chunk_offset_l):
        if chunkID not in chunkID2offset_m:
            chunkID2offset_m[chunkID] = [chunk_offset]
        else:
            chunkID2offset_m[chunkID].append(chunk_offset)

    item_n_vecs_l = train_counts
    item_n_vecs_offset_l = np.cumsum(item_n_vecs_l)
    item_n_vecs_offset_l = np.concatenate(([0], item_n_vecs_offset_l))
    vec_dim = train.shape[1]

    print("load chunk data")
    sample_vecs_l = np.array([])
    sample_item_n_vec_l = np.array([], dtype=np.uint32)
    vecsID_l = np.array([])
    for chunkID, offset_itemID_l in chunkID2offset_m.items():
        print(f"chunkID {chunkID}, n_item in chunk {len(offset_itemID_l)}")
        # item_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'encoding{chunkID}_float32.npy'))
        # item_n_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'doclens{chunkID}.npy'))
        item_vecs_l_chunk = train
        item_n_vecs_l_chunk = train_counts
        item_n_vecs_offset_chunk_l = np.cumsum(item_n_vecs_l_chunk)
        item_n_vecs_offset_chunk_l = np.concatenate(([0], item_n_vecs_offset_chunk_l))
        item_n_vecs_offset_chunk_l = np.array(item_n_vecs_offset_chunk_l, dtype=np.uint64)

        base_itemID = chunkID * DEFAULT_CHUNKSIZE
        vecsID_l_chunk = np.array([])
        item_n_vec_l_chunk = np.array([])

        for offset_itemID in offset_itemID_l:
            itemID = base_itemID + offset_itemID
            item_n_vecs = item_n_vecs_l[itemID]
            base_vecID_chunk = item_n_vecs_offset_chunk_l[offset_itemID]
            vecsID_l_chunk = np.concatenate(
                (vecsID_l_chunk, np.arange(base_vecID_chunk, base_vecID_chunk + item_n_vecs, 1))).astype(np.uint64)
            item_n_vec_l_chunk = np.concatenate((item_n_vec_l_chunk, [item_n_vecs]))

        sample_vecs_l_chunk = item_vecs_l_chunk[vecsID_l_chunk, :]
        sample_vecs_l = sample_vecs_l_chunk if len(sample_vecs_l) == 0 else np.concatenate(
            (sample_vecs_l, sample_vecs_l_chunk)).reshape(-1, vec_dim)

        vecsID_l_chunk = vecsID_l_chunk + item_n_vecs_offset_l[base_itemID]
        vecsID_l = np.concatenate(
            (vecsID_l, vecsID_l_chunk))

        sample_item_n_vec_l = np.concatenate((sample_item_n_vec_l, item_n_vec_l_chunk))

        print("finish load chunkID")

    sample_vecs_l = sample_vecs_l.reshape(-1, vec_dim)

    assert len(vecsID_l) == len(
        sample_vecs_l), f"len(vecsID_l) {len(vecsID_l)}, len(sample_vecs_l) {len(sample_vecs_l)}"
    vecsID_l = np.array(vecsID_l, dtype=np.uint64)

    sample_item_n_vec_l = sample_item_n_vec_l.astype(np.uint32)

    return sample_vecs_l, sample_item_n_vec_l, vecsID_l


def compress_into_codes(embs: np.ndarray, centroid_l: np.ndarray):
    codes = []

    centroid_l = torch.Tensor(centroid_l)
    # print(centroid_l)
    bsize = (1 << 29) // centroid_l.shape[0]
    embs = torch.from_numpy(embs)
    for batch in embs.split(bsize):
        indices = (centroid_l @ batch.T.float()).max(dim=0).indices
        indices = indices.to('cpu')
        codes.append(indices)

    return torch.cat(codes)


def item_code_in_chunk(code_l: np.ndarray, itemlen_l: np.ndarray, itemID: int):
    vecs_start_idx = int(np.sum(itemlen_l[:itemID]))
    n_item_vecs = int(itemlen_l[itemID])
    item_code = code_l[vecs_start_idx: vecs_start_idx + n_item_vecs]
    return item_code


def sample_vector(train: np.ndarray, train_counts: np.ndarray):
    # vec_dim = np.load(os.path.join(embedding_dir, 'base_embedding', f'encoding0_float32.npy')).shape[1]
    # item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
    item_n_vec_l = train_counts
    n_item = item_n_vec_l.shape[0]

    print("sample itemID for kmeans")
    sample_itemID_l = sample_itemID4kmeans(n_item=n_item)
    # DEFAULT_CHUNKSIZE = len(np.load(os.path.join(embedding_dir, 'base_embedding', f'doclens{0}.npy')))
    DEFAULT_CHUNKSIZE = len(train_counts)

    print("read sample vector from disk")
    sample_vecs_l, sample_item_n_vec_l, _ = get_sample_vecs_l(sample_itemID_l=sample_itemID_l,
                                                              DEFAULT_CHUNKSIZE=DEFAULT_CHUNKSIZE,
                                                              train=train, train_counts=train_counts)
    return sample_vecs_l, sample_item_n_vec_l


def faiss_kmeans(sample_vecs_l: np.ndarray, n_centroid: int):
    print("build kmeans")
    vec_dim = sample_vecs_l.shape[1]
    kmeans = faiss.Kmeans(vec_dim, n_centroid, niter=20, gpu=True, verbose=True, seed=123)
    kmeans.train(sample_vecs_l)

    centroids = torch.from_numpy(kmeans.centroids)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    centroids = centroids.float()
    return centroids.numpy()


def compute_assignment(train: np.ndarray, train_counts: np.ndarray, centroid_l: np.ndarray):
    n_centroid = len(centroid_l)
    code_l = torch.Tensor([])
    centroid2itemID_ivf = []
    for centroidID in range(n_centroid):
        centroid2itemID_ivf.append([])

    print("compute assignment of centroid vector")
    n_chunk = 0
    accu_itemID = 0
    for chunkID in tqdm.trange(n_chunk):
        # itemlen_l_chunk = np.load(os.path.join(base_embedding_dir, f'doclens{chunkID}.npy'))
        # item_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'encoding{chunkID}_float32.npy'))
        itemlen_l_chunk = train_counts
        item_vecs_l_chunk = train
        n_item_chunk = itemlen_l_chunk.shape[0]

        code_l_chunk = compress_into_codes(embs=item_vecs_l_chunk, centroid_l=centroid_l)
        for itemID_chunk in range(n_item_chunk):
            itemID = accu_itemID + itemID_chunk
            item_code_l = item_code_in_chunk(np.array(code_l_chunk), itemlen_l_chunk, itemID_chunk)
            for code in np.unique(item_code_l):
                assert 0 <= code < n_centroid
                centroid2itemID_ivf[code].append(itemID)
        code_l = torch.cat([code_l, code_l_chunk])
        accu_itemID += n_item_chunk

    cluster2itemID_l = np.array([itemID for itemID_l in centroid2itemID_ivf for itemID in itemID_l], dtype=np.uint32)
    cluster_n_item_l = np.array([len(itemID_l) for itemID_l in centroid2itemID_ivf], dtype=np.uint32)
    code_l = np.array(code_l, dtype=np.uint32)
    return code_l, cluster2itemID_l, cluster_n_item_l


def vq_ivf(train: np.ndarray, train_counts: np.ndarray, n_centroid: int):
    sample_vecs_l, sample_item_n_vec_l = sample_vector(train, train_counts)

    centroid_l = faiss_kmeans(sample_vecs_l=sample_vecs_l, n_centroid=n_centroid)

    code_l, cluster2itemID_l, cluster_n_item_l = compute_assignment(train=train, train_counts=train_counts,
                                                                    centroid_l=centroid_l)
    centroid_l = np.array(centroid_l, dtype=np.float32)

    return centroid_l, code_l, cluster2itemID_l, cluster_n_item_l


def compute_assignment_sq(train: np.ndarray,
                          train_counts: np.ndarray,
                          centroid_l: np.ndarray,
                          module: object,
                          cutoff_l: np.ndarray, weight_l: np.ndarray, n_bit: int):
    doclens = train_counts
    n_vec = int(np.sum(doclens))
    vq_code_l = torch.empty((n_vec), dtype=torch.int32)

    sq_ins = module.CompressResidualCode(centroid_l=centroid_l, cutoff_l=cutoff_l, weight_l=weight_l, n_bit=n_bit)
    n_val_per_vec = sq_ins.n_val_per_vec

    residual_code_l = np.empty(shape=(n_vec * n_val_per_vec), dtype=np.uint8)

    print("compute assignment of centroid vector")
    n_chunk = 1
    vq_code_offset = 0
    residual_code_offset = 0
    centroid_l_cuda = torch.Tensor(centroid_l)#.to('cuda')
    for chunkID in tqdm.trange(n_chunk):
        # itemlen_l_chunk = np.load(os.path.join(base_embedding_dir, f'doclens{chunkID}.npy'))
        itemlen_l_chunk = train_counts
        n_vec_chunk = int(np.sum(itemlen_l_chunk))
        # item_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'encoding{chunkID}_float32.npy'))
        item_vecs_l_chunk = train

        vq_code_l_chunk = compress_into_codes(embs=item_vecs_l_chunk, centroid_l=centroid_l_cuda)

        residual_code_l_chunk, = sq_ins.compute_residual_code(vec_l=item_vecs_l_chunk,
                                                            code_l=vq_code_l_chunk.numpy())
        residual_code_l_chunk = np.array(residual_code_l_chunk, dtype=np.uint8)

        vq_code_l[vq_code_offset:vq_code_offset + n_vec_chunk] = vq_code_l_chunk
        vq_code_offset += n_vec_chunk

        residual_code_l[residual_code_offset:residual_code_offset + n_vec_chunk * n_val_per_vec] = residual_code_l_chunk
        residual_code_offset += n_vec_chunk * n_val_per_vec

    vq_code_l = np.array(vq_code_l, dtype=np.uint32)
    # residual_code_l = np.array(residual_code_l, dtype=np.uint8)
    assert vq_code_l.shape[0] * n_val_per_vec == residual_code_l.shape[0]
    assert residual_code_l.ndim == 1
    return vq_code_l, residual_code_l


def faiss_kmeans_code(sample_vecs_l: np.ndarray, n_centroid: int):
    print("build kmeans")
    vec_dim = sample_vecs_l.shape[1]
    kmeans = faiss.Kmeans(vec_dim, n_centroid, niter=2, gpu=True, verbose=True, seed=123)
    kmeans.train(sample_vecs_l)

    centroids = torch.from_numpy(kmeans.centroids)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    centroids = centroids.float()
    centroids = centroids.numpy()

    code_l = compress_into_codes(embs=sample_vecs_l, centroid_l=centroids)
    return centroids, code_l


def vq_sq_ivf(train: np.ndarray, train_counts: np.ndarray, module: object, n_centroid: int, n_bit: int):
    sample_vecs_l, sample_item_n_vec_l = sample_vector(train, train_counts)

    centroid_l, sample_code_l = faiss_kmeans_code(sample_vecs_l=sample_vecs_l, n_centroid=n_centroid)

    cutoff_l, weight_l = module.compute_quantized_scalar(item_vec_l=sample_vecs_l, centroid_l=centroid_l,
                                                         code_l=sample_code_l,
                                                         n_bit=n_bit)
    vq_code_l, residual_code_l = compute_assignment_sq(train=train, train_counts=train_counts,
                                                       centroid_l=centroid_l, module=module,
                                                       cutoff_l=cutoff_l, weight_l=weight_l, n_bit=n_bit)
    centroid_l = np.array(centroid_l, dtype=np.float32)
    vq_code_l = np.array(vq_code_l, dtype=np.uint32)
    weight_l = np.array(weight_l, dtype=np.float32)
    residual_code_l = np.array(residual_code_l, dtype=np.uint8)

    return centroid_l, vq_code_l, weight_l, residual_code_l