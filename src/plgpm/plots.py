import numpy as np
import matplotlib.pyplot as plt

def reorder_by_hclust(C, method="average", metric="euclidean"):
    """
    Reorder a symmetric matrix using hierarchical clustering on rows.
    Returns (C_reordered, order_indices).
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    # Use rows as features; for symmetric similarity-like matrices,
    # clustering rows is a decent way to reveal block structure.
    D = pdist(C, metric=metric)
    Z = linkage(D, method=method)
    order = leaves_list(Z)
    return C[np.ix_(order, order)], order

def plot_coupling_heatmap(
    C,
    labels=None,
    title="MRF coupling strengths",
    log_scale=False,
    eps=1e-12,
    reorder=None,         # None or "hclust"
    vmax=None,
    figsize=(6.5, 5.5),
    show_colorbar=True,
):
    """
    C: (N,N) symmetric coupling strength matrix (diag typically 0)
    labels: list of length N (e.g., residue indices or "i:phi/psi")
    log_scale: plot log10(C+eps) to compress heavy tails
    reorder: None or "hclust"
    """
    C = np.array(C, dtype=float).copy()
    np.fill_diagonal(C, 0.0)

    order = np.arange(C.shape[0])
    if reorder == "hclust":
        C, order = reorder_by_hclust(C)
        if labels is not None:
            labels = [labels[i] for i in order]

    M = np.log10(C + eps) if log_scale else C

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(M, origin="lower", aspect="equal", vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Dihedral pair index")
    ax.set_ylabel("Dihedral pair index")

    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

    ax.grid(False)

    if show_colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log10(|J|) (a.u.)" if log_scale else "|J| strength (a.u.)")

    plt.tight_layout()
    return fig, ax, order


def plot_probability_heatmap(
    P,
    title="Joint probability",
    xlabel="State of variable b",
    ylabel="State of variable a",
    cmap="viridis",
    vmin=0.0,
    vmax=None,
    figsize=(4.5, 4.0),
    show_colorbar=True,
):
    """
    Plot a 2D probability table as a heatmap.
    """
    P = np.asarray(P, dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(P, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Probability")

    plt.tight_layout()
    return fig, ax
