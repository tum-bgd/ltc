import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def plot_one(X, y, ax, name, s):
    cmap = plt.cm.tab20
    unique_labels = np.unique(y)
    for label in unique_labels:
        if label == -1:
            color = "black"
        else:
            floor, mod = divmod(int(label), 10)
            color = cmap(int(2 * mod + floor) % 20)
        label_points = X[y == label]
        ax.scatter(label_points[:, 0], label_points[:, 1], color=color, s=s)
    ax.axis("off")
    ax.set_title(name)


def plot_all(X, y, y_ltc, lifetime, y_dbs, s=10):
    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    plot_one(X, y, ax1, "True labels", s)
    plot_one(X, y_ltc, ax3, "LTC result", s)
    plot_one(X, y_dbs, ax4, "DBS result", s)

    ax2.scatter(X[:, 0], X[:, 1], s=s, c=lifetime, cmap=plt.cm.jet)
    ax2.axis("off")
    ax2.set_title("Lifetime")

    plt.tight_layout()
    plt.show()


def plot_comparison(
    noise_levels,
    x_label,
    y_label,
    ari_ltc_means,
    ari_ltc_stds,
    ari_dbs_means,
    ari_dbs_stds,
):
    plt.figure(figsize=(6, 4))
    plt.plot(noise_levels, ari_ltc_means, marker="o", label="LTC")
    plt.fill_between(
        noise_levels,
        ari_ltc_means - ari_ltc_stds,
        ari_ltc_means + ari_ltc_stds,
        alpha=0.2,
    )
    plt.plot(noise_levels, ari_dbs_means, marker="x", label="DBS")
    plt.fill_between(
        noise_levels,
        ari_dbs_means - ari_dbs_stds,
        ari_dbs_means + ari_dbs_stds,
        alpha=0.2,
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.show()


def normalize_data(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-10)


def eval_clustering(y_true, y_pred):
    from sklearn.metrics import adjusted_rand_score, v_measure_score

    ari = adjusted_rand_score(y_true, y_pred)
    vme = v_measure_score(y_true, y_pred)
    uniques = np.unique(y_pred)
    num_clusters = uniques.size
    return ari, vme, num_clusters


def make_two_blobs(r, n, d, noise):
    theta = 2 * np.pi * np.random.rand(2 * n)
    rad = r * np.sqrt(np.random.rand(2 * n))
    c1 = np.array([0.0, 0.0])
    c2 = np.array([d, 0.0])

    centers = np.vstack(
        [np.repeat(c1[None, :], n, axis=0), np.repeat(c2[None, :], n, axis=0)]
    )

    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    points = np.column_stack([x, y]) + centers

    if noise > 0.0:
        points += np.random.normal(scale=noise, size=points.shape)

    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    return points, labels
