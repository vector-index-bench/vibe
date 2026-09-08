# Contributing a new method

## Scope

VIBE benchmarks **algorithms**, with one representative implementation per algorithm by default. Before adding a method, check the [existing algorithms](../README.md#algorithms):

- New algorithms are welcome.
- Additional implementations of an existing algorithm should introduce a significant algorithmic addition, such as quantization.
- Vector databases and database service integrations are out of scope.

## Integration

Under `vibe/algorithms/<algorithm>/`, add:
- `image.def` container definition
- `module.py` adapter
- `config.yml` parameter grid

See e.g. [hnswlib](../vibe/algorithms/hnswlib/) for a small working example.

### 1. Build the image

Create `image.def`. Install the method and its dependencies here, pinning the method to a version or a Git commit:

```def
Bootstrap: localimage
From: base.sif

%post
  apt-get install -y libomp-dev
  pip install --no-cache-dir example-ann==1.2.3
```

Build it with the directory name:

```sh
./install.sh --algorithm example
```

Use `--force` after changing `image.def`.

### 2. Implement the adapter

Subclass `BaseANN` in `module.py`:

```python
import numpy as np
from example_ann import Index

from ..base.module import BaseANN


class ExampleANN(BaseANN):
    def __init__(self, metric, build_budget, graph_degree):
        self.metric = metric
        self.build_budget = build_budget
        self.graph_degree = graph_degree

    def fit(self, X):
        self.index = Index(
            X, metric=self.metric, graph_degree=self.graph_degree, threads=1
        )
        self.index.build(self.build_budget)

    def set_query_arguments(self, search_budget):
        self.search_budget = search_budget

    def query(self, vector, n):
        ids = self.index.search(vector, n, self.search_budget)
        return np.asarray(ids, dtype=np.int64)

    def __str__(self):
        return (
            f"ExampleANN(build={self.build_budget}, degree={self.graph_degree}, "
            f"search={self.search_budget})"
        )
```

The important adapter methods are:

- `__init__(*build_args)` stores build parameters; do not build the index yet.
- `fit(X)` builds the index using the corpus vectors stored in a NumPy array `X`.  Do not modify `X` in place. All indexes are built using a single thread.
- `set_query_arguments(*query_args)` applies one query-time hyperparameter combination.
- `query(vector, n)` returns at most `n` unique row IDs from `X`, not distances.
- `__str__()` records every build and query parameter affecting the result.

GPU methods should implement `batch_query(X, n)` and `get_batch_results()`.

OOD methods that leverage query samples should implement `fit_ood(X_train, X_learn, X_learn_neighbors)`, where `X_learn` contains the query samples and `X_learn_neighbors` contains the 100 nearest neighbors for each query sample.

### 3. Configure the hyperparameters

Create `config.yml`:

```yaml
float:
  any:
  - name: example
    module: vibe.algorithms.example
    constructor: ExampleANN
    singularity_image: example
    disabled: false
    gpu: false
    ood: false
    base_args: ['@metric']
    run_groups:
      default:
        args:
          build_budget: [100, 200]
          graph_degree: [16, 32]
        query_args:
          search_budget: [20, 40, 80, 160]
```

Fill in `name`, `module`, `constructor`, and `singularity_image` (usually just the name of the method for each).

The outer key selects the supported datatype: `float` for regular float vectors, `int8`, `uint8`, or `binary` for quantized vectors. The next key is a supported distance (`euclidean`, `cosine`, `ip`, `normalized`, or `hamming`) or `any` if the method supports all distances.

`args` and `query_args` are Cartesian-product grids passed positionally in YAML order. Put only index-building parameters in `args`. Generally, it's best to keep the number of indexing combinations below 20 and query combinations below 60.

Set `gpu: true` for GPU methods. They run only with `run.py --gpu` and must implement the batch-query methods described above. Set `ood: true` when the method uses the learning queries and neighbors supplied by out-of-distribution datasets; VIBE then calls `fit_ood` when that data is available. Both keys default to `false` and may be omitted.

### 4. Update the README and open a pull request

Add the method and pinned implementation version to the [README](../README.md), then open a pull request.
