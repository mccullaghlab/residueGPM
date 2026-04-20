# plgpm

Pseudo-likelihood Generalized Potts Model for discrete multistate data.

## Installation

```bash
pip install plgpm
```

## Quick Start

```python
from plgpm import PLGPM

K_list = [12, 12, 12]
model = PLGPM(K_list)
model.fit(S_train)

pll = model.score_pseudologlik(S_test)
samples = model.sample(n_samples=10000, burn=1000, thin=5)
C = model.coupling_matrix()
```
## What this package does

`plgpm` fits node-conditional multinomial logistic models for discrete graphical modeling and provides coupling summaries, Gibbs sampling, and evaluation utilities.


## Tests

You do not need many tests at first, but you do need a few. I would include:

- fit runs on a tiny synthetic dataset
- sampled states are all within valid bounds
- `conditional_probs()` sums to 1
- coupling matrix is symmetric with zero diagonal
- evaluation output has expected fields

That will make your package much safer to release.

## GitHub Actions CI

A minimal CI workflow can run tests on pushes and pull requests. GitHub’s official docs show standard Python testing workflows. :contentReference[oaicite:7]{index=7}

`.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install package and test dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]

      - name: Run tests
        run: pytest
```
