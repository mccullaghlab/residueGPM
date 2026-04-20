import numpy as np
import matplotlib.pyplot as plt


def mutual_information_from_joint_counts(counts, pseudocount=0.0):
    """
    Compute MI (in nats) from a joint count matrix.

    Parameters
    ----------
    counts : ndarray, shape (Ki, Kj)
        Joint counts for two discrete variables.
    pseudocount : float
        Small value added to every cell for smoothing.

    Returns
    -------
    mi : float
        Mutual information in nats.
    """
    counts = counts.astype(float)
    if pseudocount > 0:
        counts = counts + pseudocount

    total = counts.sum()
    if total <= 0:
        return 0.0

    pxy = counts / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    expected = px @ py

    mask = pxy > 0
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask]))
    return mi


def pair_joint_counts(x, y, Ki=None, Kj=None):
    """
    Build joint count matrix for two integer-valued discrete arrays.

    Parameters
    ----------
    x, y : ndarray, shape (n_samples,)
        Integer state labels.
    Ki, Kj : int or None
        Number of states for x and y. If None, inferred from max label + 1.

    Returns
    -------
    counts : ndarray, shape (Ki, Kj)
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)

    if Ki is None:
        Ki = int(x.max()) + 1
    if Kj is None:
        Kj = int(y.max()) + 1

    counts = np.zeros((Ki, Kj), dtype=np.int64)
    np.add.at(counts, (x, y), 1)
    return counts


def compute_mi_matrix(states, K_list=None, pseudocount=0.0):
    """
    Compute residue-residue MI matrix from discrete microstate assignments.

    Parameters
    ----------
    states : ndarray, shape (n_frames, n_residues)
        Integer-valued microstate labels for each residue.
    K_list : list[int] or None
        Number of states for each residue. If None, inferred from data.
    pseudocount : float
        Small pseudocount added to joint count tables.

    Returns
    -------
    mi_mat : ndarray, shape (n_residues, n_residues)
        Symmetric MI matrix in nats.
    """
    states = np.asarray(states, dtype=int)
    n_frames, n_residues = states.shape

    if K_list is None:
        K_list = [int(states[:, i].max()) + 1 for i in range(n_residues)]

    mi_mat = np.zeros((n_residues, n_residues), dtype=float)

    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            counts = pair_joint_counts(states[:, i], states[:, j],
                                       Ki=K_list[i], Kj=K_list[j])
            mi = mutual_information_from_joint_counts(counts, pseudocount=pseudocount)
            mi_mat[i, j] = mi
            mi_mat[j, i] = mi

    return mi_mat


def upper_triangle_values(mat, k=1):
    """
    Return upper-triangle values of a square matrix.
    """
    iu = np.triu_indices_from(mat, k=k)
    return mat[iu]


def plot_mi_comparison(mi_md, mi_mrf, title_prefix="", cmap="viridis"):
    """
    Plot MI matrices and their difference.
    """
    diff = mi_mrf - mi_md
    vmax = max(mi_md.max(), mi_mrf.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(mi_md, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
    axes[0].set_title(f"{title_prefix}MD MI")
    axes[0].set_xlabel("Residue index")
    axes[0].set_ylabel("Residue index")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="MI (nats)")

    im1 = axes[1].imshow(mi_mrf, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
    axes[1].set_title(f"{title_prefix}MRF MI")
    axes[1].set_xlabel("Residue index")
    axes[1].set_ylabel("Residue index")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="MI (nats)")

    absmax = np.max(np.abs(diff))
    im2 = axes[2].imshow(diff, origin="lower", cmap="coolwarm",
                         vmin=-absmax, vmax=absmax)
    axes[2].set_title(f"{title_prefix}MRF - MD")
    axes[2].set_xlabel("Residue index")
    axes[2].set_ylabel("Residue index")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="ΔMI (nats)")

    plt.show()


def plot_mi_scatter(mi_md, mi_mrf, title="MRF vs MD pairwise MI"):
    """
    Scatter plot of pairwise MI values.
    """
    x = upper_triangle_values(mi_md, k=1)
    y = upper_triangle_values(mi_mrf, k=1)

    maxval = max(x.max(), y.max()) if len(x) > 0 else 1.0

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, alpha=0.7)
    plt.plot([0, maxval], [0, maxval], 'k--', linewidth=1)
    plt.xlabel("MD pairwise MI (nats)")
    plt.ylabel("MRF pairwise MI (nats)")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def summarize_mi_agreement(mi_md, mi_mrf):
    """
    Numerical summary of agreement between MI matrices.
    """
    x = upper_triangle_values(mi_md, k=1)
    y = upper_triangle_values(mi_mrf, k=1)

    diff = y - x
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    if np.std(x) > 0 and np.std(y) > 0:
        corr = np.corrcoef(x, y)[0, 1]
    else:
        corr = np.nan

    return {
        "mean_md_mi": float(np.mean(x)),
        "mean_mrf_mi": float(np.mean(y)),
        "sum_md_mi": float(np.sum(x)),
        "sum_mrf_mi": float(np.sum(y)),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson_r": float(corr),
        "max_abs_error": float(np.max(np.abs(diff))),
    }

