<h1 align="center">VIBE</h1>
<div align="center">
Vector Index Benchmark for Embeddings (VIBE)
</div>
<br/>

---

- Modern vector index benchmark with embedding data sets
- Includes data sets for both in-distribution and out-out-distribution settings
- Includes the most comprehensive collection of state-of-the-art vector search algorithms
- Support for quantized data sets in both 8-bit integer and binary precision
- Support for GPU algorithms
- Support for HPC environments with SLURM

## Results

The current VIBE results can be viewed at:
https://vector-index-bench.github.io

### Data sets

| Data Source | Model | Type | n | d | Distance |
|-------------|-------|------|-------|-----|----------|
| AGNews | MXBAI | Text | 769,382 | 1024 | euclidean |
| arXiv abstracts | Nomic Text | Text | 1,344,643 | 768 | any |
| Google Q&A | DistilRoBERTa | Text | 1,475,024 | 768 | any |
| ImageNet | CLiP | Image | 1,281,167 | 512 | any |
| Landmark | Nomic Vision | Image | 760,757 | 768 | any |
| Yahoo Answers | MiniLM | Text | 677,305 | 384 | any |
| CelebA | ResNet | Image | 201,599 | 2048 | cosine |
| CCNews | Nomic Text | Text | 495,328 | 768 | any |
| CodeSearchNet | Jina | Code | 1,374,067 | 768 | cosine |
| - | GloVe | Word | 1,192,514 | 200 | cosine |
| Landmark | DINOv2 | Image | 760,757 | 768 | cosine |
| Simple Wiki | OpenAI | Text | 260,372 | 3072 | any |

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

### Building library images

Building an image for a library can be done using `install.sh`. It can be used to either build an image for a single library (`./install.sh --algorithm faiss`) or images for all available libraries (`./install.sh`). Additionally, it takes an argument `--build-dir` that specifies the temporary build directory (e.g. to speed up the build in a cluster environment, you can set the build directory to a location on an SSD while the project files are on slower storage medium).
