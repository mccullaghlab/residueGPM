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

## Streamlined post-fit analysis

```python
from plgpm import PLGPM, mi, plots

model = PLGPM(obj.components).fit(micro_states)
analysis = model.analyze(
    data_states=micro_states,
    pair=(0, 1),  # optional residue pair for joint probabilities
    sample_kwargs={"n_samples": 50_000, "burn": 5_000, "thin": 10},
)

# Couplings
plots.plot_coupling_heatmap(analysis.coupling)

# Pair joint probabilities (model vs empirical)
plots.plot_probability_heatmap(analysis.pair["prob_model"], title="Model P(x0, x1)")
plots.plot_probability_heatmap(analysis.pair["prob_data"], title="MD P(x0, x1)")

# Pairwise MI agreement (model vs empirical)
mi.plot_mi_comparison(analysis.mi_data, analysis.mi_model, title_prefix="Dialanine Tripeptide: ")
mi.plot_mi_scatter(analysis.mi_data, analysis.mi_model, title="Dialanine Tripeptide: MRF vs MD pairwise MI")
print(analysis.mi_summary)
```
## What this package does

`plgpm` fits node-conditional multinomial logistic models for discrete graphical modeling and provides coupling summaries, Gibbs sampling, and evaluation utilities.


