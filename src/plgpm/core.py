from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple, Dict, Any

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression


ArrayLikeInt = np.ndarray


@dataclass
class PLGPMConfig:
    C: float = 0.1
    max_iter: int = 1500
    tol: float = 1e-3
    l1_ratio: float = 0.5
    penalty: str = "elasticnet"
    solver: str = "saga"
    n_jobs: int = 4
    random_state: Optional[int] = None
    verbose: bool = True


@dataclass
class EvaluationResult:
    pll_gpm: float
    pll_ind: float
    pll_gain_gpm_minus_ind: float
    pair_results: Dict[Tuple[int, int], Dict[str, float]]
    full_joint: Optional[Dict[str, Any]]
    n_gibbs_samples: int
    pairs: List[Tuple[int, int]]
    N: int


class PLGPM:
    """
    Pseudo-likelihood Generalized Potts Model for discrete multivariate data.

    Parameters
    ----------
    K_list
        Number of states for each variable.
    config
        Hyperparameter/configuration object.
    """

    def __init__(self, K_list: Sequence[int], config: Optional[PLGPMConfig] = None):
        self.K_list = list(map(int, K_list))
        self.config = PLGPMConfig() if config is None else config

        self.models: Optional[List[Tuple[Optional[LogisticRegression], Optional[np.ndarray]]]] = None
        self.slices: Optional[List[slice]] = None
        self.is_fitted: bool = False

    # -------------------------------------------------------------------------
    # validation
    # -------------------------------------------------------------------------

    def _validate_S(self, S: ArrayLikeInt) -> np.ndarray:
        S = np.asarray(S)
        if S.ndim != 2:
            raise ValueError("S must be a 2D array of shape (T, N).")

        T, N = S.shape
        if N != len(self.K_list):
            raise ValueError(
                f"S has {N} columns but len(K_list) = {len(self.K_list)}."
            )

        if not np.issubdtype(S.dtype, np.integer):
            raise ValueError("S must contain integer state labels.")

        for i, Ki in enumerate(self.K_list):
            if np.any(S[:, i] < 0) or np.any(S[:, i] >= Ki):
                raise ValueError(
                    f"Column {i} contains values outside 0..{Ki - 1}."
                )

        return S.astype(np.int64, copy=False)

    def _check_fitted(self) -> None:
        if not self.is_fitted or self.models is None or self.slices is None:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")

    # -------------------------------------------------------------------------
    # one-hot construction
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_onehot_sparse(S: np.ndarray, K_list: Sequence[int]) -> Tuple[sp.csr_matrix, List[slice]]:
        T, N = S.shape
        D = int(np.sum(K_list))

        rows = np.arange(T)
        cols = []
        slices = []
        col0 = 0

        for i, Ki in enumerate(K_list):
            sl = slice(col0, col0 + Ki)
            slices.append(sl)
            cols.append(col0 + S[:, i])
            col0 += Ki

        cols = np.stack(cols, axis=1).reshape(-1)
        rows_rep = np.repeat(rows, N)
        data = np.ones_like(cols, dtype=np.float32)

        X = sp.csr_matrix((data, (rows_rep, cols)), shape=(T, D), dtype=np.float32)
        return X, slices

    # -------------------------------------------------------------------------
    # fitting
    # -------------------------------------------------------------------------

    def _fit_one_node(
        self,
        i: int,
        X_all: sp.csr_matrix,
        slices: List[slice],
        S: np.ndarray,
    ) -> Tuple[Optional[LogisticRegression], Optional[np.ndarray]]:
        if self.config.verbose:
            print(f"Fitting node {i + 1}/{len(self.K_list)}", flush=True)

        Ki = self.K_list[i]
        if Ki <= 1:
            return (None, None)

        mask = np.ones(X_all.shape[1], dtype=bool)
        mask[slices[i]] = False

        X_i = X_all[:, mask]
        y_i = S[:, i]

        clf = LogisticRegression(
            penalty=self.config.penalty,
            l1_ratio=self.config.l1_ratio,
            solver=self.config.solver,
            C=self.config.C,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            n_jobs=1,  # important when using outer joblib parallelism
            random_state=self.config.random_state,
        )
        clf.fit(X_i, y_i)
        return (clf, mask)

    def fit(self, S: ArrayLikeInt) -> "PLGPM":
        S = self._validate_S(S)
        X_all, slices = self._build_onehot_sparse(S, self.K_list)

        models = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._fit_one_node)(i, X_all, slices, S)
            for i in range(len(self.K_list))
        )

        self.models = models
        self.slices = slices
        self.is_fitted = True
        return self

    # -------------------------------------------------------------------------
    # conditional probabilities and sampling
    # -------------------------------------------------------------------------

    @staticmethod
    def _onehot_row_from_state(
        state_vec: np.ndarray,
        slices: Sequence[slice],
        K_list: Sequence[int],
    ) -> sp.csr_matrix:
        N = len(K_list)
        cols = []
        for j in range(N):
            sl = slices[j]
            cols.append(sl.start + int(state_vec[j]))

        data = np.ones(N, dtype=np.float32)
        rows = np.zeros(N, dtype=np.int64)
        X = sp.csr_matrix(
            (data, (rows, np.array(cols))),
            shape=(1, slices[-1].stop),
            dtype=np.float32,
        )
        return X

    def conditional_probs(self, i: int, state_vec: Sequence[int]) -> np.ndarray:
        self._check_fitted()
        state_vec = np.asarray(state_vec, dtype=np.int64)

        clf, mask = self.models[i]
        Ki = self.K_list[i]

        if clf is None or Ki <= 1:
            p = np.zeros(Ki, dtype=float)
            if Ki > 0:
                p[0] = 1.0
            return p

        X_full = self._onehot_row_from_state(state_vec, self.slices, self.K_list)
        X_red = X_full[:, mask]
        p = clf.predict_proba(X_red).reshape(-1)
        p = np.clip(p, 1e-15, 1.0)
        p /= p.sum()
        return p

    def sample(
        self,
        n_samples: int = 200_000,
        burn: int = 10_000,
        thin: int = 10,
        init_state: Optional[Sequence[int]] = None,
        rng: Optional[np.random.Generator] = None,
        sweep_order: str = "random",
    ) -> np.ndarray:
        self._check_fitted()

        rng = np.random.default_rng() if rng is None else rng
        N = len(self.K_list)

        if init_state is None:
            state = np.array([rng.integers(0, self.K_list[i]) for i in range(N)], dtype=np.int64)
        else:
            state = np.asarray(init_state, dtype=np.int64).copy()

        kept = []
        total_steps = burn + n_samples * thin

        for t in range(total_steps):
            if sweep_order == "random":
                order = rng.permutation(N)
            elif sweep_order == "fixed":
                order = np.arange(N)
            else:
                raise ValueError("sweep_order must be 'random' or 'fixed'.")

            for i in order:
                Ki = self.K_list[i]
                if Ki <= 1:
                    continue
                p = self.conditional_probs(i, state)
                state[i] = rng.choice(Ki, p=p)

            if t >= burn and ((t - burn) % thin == 0):
                kept.append(state.copy())

        return np.asarray(kept, dtype=np.int64)

    # -------------------------------------------------------------------------
    # scoring
    # -------------------------------------------------------------------------

    def score_pseudologlik(self, S: ArrayLikeInt) -> float:
        self._check_fitted()
        S = self._validate_S(S)

        T, N = S.shape
        ll = 0.0
        for t in range(T):
            state = S[t]
            for i in range(N):
                p = self.conditional_probs(i, state)
                ll += np.log(p[int(state[i])] + 1e-15)
        return ll / (T * N)

    @staticmethod
    def fit_independence_baseline(S: np.ndarray, K_list: Sequence[int]) -> List[np.ndarray]:
        ps = []
        for i, Ki in enumerate(K_list):
            c = np.bincount(S[:, i], minlength=Ki).astype(float)
            p = c / c.sum()
            ps.append(p)
        return ps

    @staticmethod
    def score_independence(S: np.ndarray, ps: Sequence[np.ndarray]) -> float:
        T, N = S.shape
        ll = 0.0
        for t in range(T):
            for i in range(N):
                ll += np.log(ps[i][int(S[t, i])] + 1e-15)
        return ll / (T * N)

    # -------------------------------------------------------------------------
    # couplings
    # -------------------------------------------------------------------------

    def coupling_matrix(self) -> np.ndarray:
        self._check_fitted()
        N = len(self.K_list)
        full_cols = [np.arange(sl.start, sl.stop) for sl in self.slices]

        C = np.zeros((N, N), dtype=float)

        for i, (clf, mask) in enumerate(self.models):
            if clf is None:
                continue

            W = clf.coef_
            reduced_to_full = np.where(mask)[0]

            for j in range(N):
                if j == i or self.K_list[j] <= 1:
                    continue

                cols_full_j = full_cols[j]
                cols_reduced = np.where(np.isin(reduced_to_full, cols_full_j))[0]
                W_block = W[:, cols_reduced]
                C[i, j] = np.linalg.norm(W_block, ord="fro")

        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)
        return C

    # -------------------------------------------------------------------------
    # information-theoretic utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def joint_counts(S: np.ndarray, K_list: Sequence[int], a: int, b: int) -> np.ndarray:
        Ka, Kb = K_list[a], K_list[b]
        C = np.zeros((Ka, Kb), dtype=np.int64)
        np.add.at(C, (S[:, a], S[:, b]), 1)
        return C

    @staticmethod
    def joint_counts_subset(S: np.ndarray, K_list: Sequence[int], vars_: Sequence[int]) -> np.ndarray:
        vars_ = tuple(vars_)
        Ks = [K_list[i] for i in vars_]
        C = np.zeros(Ks, dtype=np.int64)
        idx = tuple(S[:, i] for i in vars_)
        np.add.at(C, idx, 1)
        return C

    @staticmethod
    def prob_from_counts(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        C = C.astype(np.float64)
        return C / (C.sum() + eps)

    @classmethod
    def mutual_information_empirical(cls, S: np.ndarray, K_list: Sequence[int], a: int, b: int, eps: float = 1e-12) -> float:
        C = cls.joint_counts(S, K_list, a, b).astype(np.float64)
        P = C / (C.sum() + eps)
        Pa = P.sum(axis=1, keepdims=True)
        Pb = P.sum(axis=0, keepdims=True)
        return float(np.sum(P * (np.log(P + eps) - np.log(Pa + eps) - np.log(Pb + eps))))

    @classmethod
    def kl_divergence_from_counts(cls, CP: np.ndarray, CQ: np.ndarray, eps: float = 1e-12) -> float:
        P = cls.prob_from_counts(CP, eps=eps)
        Q = cls.prob_from_counts(CQ, eps=eps)
        return float(np.sum(P * (np.log(P + eps) - np.log(Q + eps))))

    @classmethod
    def total_variation_from_counts(cls, CP: np.ndarray, CQ: np.ndarray, eps: float = 1e-12) -> float:
        P = cls.prob_from_counts(CP, eps=eps)
        Q = cls.prob_from_counts(CQ, eps=eps)
        return float(0.5 * np.sum(np.abs(P - Q)))

    @classmethod
    def permutation_test_mi(
        cls,
        S: np.ndarray,
        K_list: Sequence[int],
        a: int,
        b: int,
        n_perm: int = 2000,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, np.ndarray, float]:
        rng = np.random.default_rng() if rng is None else rng
        mi_obs = cls.mutual_information_empirical(S, K_list, a=a, b=b)

        mi_perm = np.empty(n_perm, dtype=float)
        Sb = S[:, b].copy()
        for r in range(n_perm):
            rng.shuffle(Sb)
            S_perm = S.copy()
            S_perm[:, b] = Sb
            mi_perm[r] = cls.mutual_information_empirical(S_perm, K_list, a=a, b=b)

        p = (1.0 + np.sum(mi_perm >= mi_obs)) / (n_perm + 1.0)
        return float(mi_obs), mi_perm, float(p)

    @staticmethod
    def all_pairs(N: int) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

    # -------------------------------------------------------------------------
    # evaluation
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        S: ArrayLikeInt,
        pairs: Optional[List[Tuple[int, int]]] = None,
        do_full_joint: bool = True,
        gibbs_kwargs: Optional[Dict[str, Any]] = None,
        n_perm: int = 500,
        test_split: float = 0.2,
        rng: Optional[np.random.Generator] = None,
    ) -> EvaluationResult:
        self._check_fitted()
        S = self._validate_S(S)

        rng = np.random.default_rng() if rng is None else rng
        gibbs_kwargs = {} if gibbs_kwargs is None else dict(gibbs_kwargs)

        T, N = S.shape
        if pairs is None:
            pairs = self.all_pairs(N)

        samples = self.sample(rng=rng, **gibbs_kwargs)

        idx = rng.permutation(T)
        n_te = int(np.round(test_split * T))
        te = S[idx[:n_te]]
        tr = S[idx[n_te:]]

        ps = self.fit_independence_baseline(tr, self.K_list)
        pll_gpm = self.score_pseudologlik(te)
        pll_ind = self.score_independence(te, ps)
        gain = pll_gpm - pll_ind

        pair_results = {}
        for (a, b) in pairs:
            mi_data = self.mutual_information_empirical(S, self.K_list, a=a, b=b)
            _, mi_perm, p_perm = self.permutation_test_mi(
                S, self.K_list, a=a, b=b, n_perm=n_perm, rng=rng
            )
            mi_gpm = self.mutual_information_empirical(samples, self.K_list, a=a, b=b)

            C_data = self.joint_counts(S, self.K_list, a=a, b=b)
            C_gpm = self.joint_counts(samples, self.K_list, a=a, b=b)

            pair_results[(a, b)] = {
                "mi_data_nats": float(mi_data),
                "mi_gpm_nats": float(mi_gpm),
                "mi_perm_p_value": float(p_perm),
                "mi_perm_mean": float(np.mean(mi_perm)),
                "kl_data||gpm_pair": self.kl_divergence_from_counts(C_data, C_gpm),
                "kl_gpm||data_pair": self.kl_divergence_from_counts(C_gpm, C_data),
                "tv_pair": self.total_variation_from_counts(C_data, C_gpm),
            }

        full_joint = None
        if do_full_joint:
            vars_ = tuple(range(N))
            C_data_full = self.joint_counts_subset(S, self.K_list, vars_)
            C_gpm_full = self.joint_counts_subset(samples, self.K_list, vars_)
            full_joint = {
                "kl_data||gpm_full": self.kl_divergence_from_counts(C_data_full, C_gpm_full),
                "kl_gpm||data_full": self.kl_divergence_from_counts(C_gpm_full, C_data_full),
                "tv_full": self.total_variation_from_counts(C_data_full, C_gpm_full),
                "vars": vars_,
                "shape": tuple(C_data_full.shape),
            }

        return EvaluationResult(
            pll_gpm=float(pll_gpm),
            pll_ind=float(pll_ind),
            pll_gain_gpm_minus_ind=float(gain),
            pair_results=pair_results,
            full_joint=full_joint,
            n_gibbs_samples=int(samples.shape[0]),
            pairs=pairs,
            N=int(N),
        )


