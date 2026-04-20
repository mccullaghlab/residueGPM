import numpy as np

from plgpm import PLGPM


def _toy_states(seed: int = 0, n_samples: int = 120, n_vars: int = 4, n_states: int = 3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_states, size=(n_samples, n_vars), dtype=np.int64)


def _fit_toy_model():
    S = _toy_states()
    model = PLGPM([3, 3, 3, 3])
    model.fit(S)
    return model, S


def test_fit_runs_on_tiny_dataset():
    model, _ = _fit_toy_model()
    assert model.is_fitted is True
    assert model.models is not None
    assert len(model.models) == 4


def test_sampled_states_stay_within_bounds():
    model, _ = _fit_toy_model()
    samples = model.sample(n_samples=100, burn=20, thin=1)

    assert samples.shape == (100, 4)
    for i, Ki in enumerate(model.K_list):
        assert np.all(samples[:, i] >= 0)
        assert np.all(samples[:, i] < Ki)


def test_conditional_probs_sum_to_one():
    model, S = _fit_toy_model()
    p = model.conditional_probs(0, S[0])

    assert p.shape == (3,)
    assert np.isclose(np.sum(p), 1.0)
    assert np.all(p >= 0.0)


def test_coupling_matrix_is_symmetric_with_zero_diagonal():
    model, _ = _fit_toy_model()
    C = model.coupling_matrix()

    assert C.shape == (4, 4)
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 0.0)


def test_evaluation_has_expected_fields():
    model, S = _fit_toy_model()
    result = model.evaluate(
        S,
        pairs=[(0, 1)],
        do_full_joint=False,
        n_perm=5,
        gibbs_kwargs={"n_samples": 120, "burn": 20, "thin": 1},
        rng=np.random.default_rng(123),
    )

    assert isinstance(result.pll_gpm, float)
    assert isinstance(result.pll_ind, float)
    assert isinstance(result.pll_gain_gpm_minus_ind, float)
    assert result.pairs == [(0, 1)]
    assert (0, 1) in result.pair_results
    assert result.full_joint is None
    assert result.n_gibbs_samples == 120
