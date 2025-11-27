import numpy as np


def plot_labels(X, y, s=10):
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.rcParams.update(mpl.rcParamsDefault)
    cmap = plt.cm.tab20
    plt.figure(figsize=(5, 5))
    unique_labels = np.unique(y)
    for label in unique_labels:
        if label == -1:
            color = "black"
        else:
            floor = int(label) // 10
            mod = int(label) % 10
            color = cmap(int(2 * mod + floor) % 20)
        label_points = X[y == label]
        plt.scatter(label_points[:, 0], label_points[:, 1], color=color, s=s)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_lifetime(X, lifetime, s=10):
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], s=s, c=lifetime, cmap=plt.cm.jet)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def normalize_data(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-10)
    return X_norm


def eval_clustering(y_true, y_pred):
    from sklearn.metrics import adjusted_rand_score, v_measure_score

    ari = adjusted_rand_score(y_true, y_pred)
    vme = v_measure_score(y_true, y_pred)
    uniques = np.unique(y_pred[y_pred >= 0])
    num_clusters = uniques.size
    return ari, vme, num_clusters


def make_two_blobs(r, n, noise):
    theta = 2 * np.pi * np.random.rand(2 * n)
    rad = r * np.sqrt(np.random.rand(2 * n))
    c1 = np.array([0.0, 0.0])
    c2 = np.array([2.5 * r, 0.0])

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
