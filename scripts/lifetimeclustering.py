"""
Lifetime Clustering (LTC): Efficient and Robust Topology-Based Clustering
"""

from collections import deque
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


class LifetimeClustering(object):
    """Lifetime Clustering Algorithm.

    Reference
    ----------
    J M Zollner, B Teuscher, W Mansour, and M Werner, "Efficient and Robust
    Topology-Based Clustering"
    """

    def __init__(self, eps):
        """Initialize Lifetime Clustering .

        Parameters
        ----------
        eps: float
            The maximum distance between two data points for one to be considered
            as in the neighborhood of the other.
        """
        eps = float(eps)
        if eps <= 0.0:
            raise ValueError("Epsilon must be positive")

        self.eps = eps
        self.kdtree = None
        self.indptr = None
        self.indices = None

    def index_points(self):
        """Build a KD-tree."""
        self.kdtree = cKDTree(self.data)

    def build_radius_csr(self):
        """Construct fixed radius neighbor graph as sparse (CSR) matrix."""
        n = self.len
        coo = self.kdtree.sparse_distance_matrix(
            self.kdtree, max_distance=self.eps, output_type="coo_matrix"
        )
        mask = coo.row != coo.col
        rows = coo.row[mask]
        cols = coo.col[mask]
        data = np.ones_like(rows, dtype=np.int8)
        mat = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        self.indptr = mat.indptr.astype(np.intp, copy=False)
        self.indices = mat.indices.astype(np.intp, copy=False)

    def contraction(self):
        """Compute lifetime."""
        n = self.len
        self.build_radius_csr()
        indptr, indices = self.indptr, self.indices

        self.lifetime = np.full(n, -1, dtype=np.int32)
        self.start_degree = (indptr[1:] - indptr[:-1]).astype(np.int32, copy=False)
        self.degree = self.start_degree.copy()
        self.sorted_ids = np.empty(n, dtype=np.int32)

        max_deg = int(self.degree.max(initial=0))
        buckets = [deque() for _ in range(max_deg + 1)]
        for i, d in enumerate(self.degree):
            buckets[int(d)].append(i)

        rpos = n - 1
        r = 1
        current_min_deg = 0

        while current_min_deg <= max_deg:
            while current_min_deg <= max_deg and not buckets[current_min_deg]:
                current_min_deg += 1

            if current_min_deg > max_deg:
                break

            current_bucket = buckets[current_min_deg]
            if not current_bucket:
                continue

            r_ids = []
            while current_bucket:
                idx = current_bucket.popleft()
                if self.lifetime[idx] == -1:
                    r_ids.append(idx)

            if not r_ids:
                continue

            unique_nb = np.zeros(n, dtype=bool)
            for idx in r_ids:
                self.lifetime[idx] = r
                self.sorted_ids[rpos] = idx
                rpos -= 1
                unique_nb[indices[indptr[idx] : indptr[idx + 1]]] = True

            unique_nb = np.where(unique_nb)[0]

            alive = self.lifetime[unique_nb] == -1
            if np.any(alive):
                nb = unique_nb[alive]
                old_deg = self.degree[nb]
                pos = old_deg > 0
                nb = nb[pos]
                if nb.size:
                    new_deg = old_deg[pos] - 1
                    self.degree[nb] = new_deg

                    order = np.argsort(new_deg)
                    nd_sorted = new_deg[order]
                    nb_sorted = nb[order]
                    changes = np.flatnonzero(
                        np.r_[True, nd_sorted[1:] != nd_sorted[:-1], True]
                    )
                    for a, b in zip(changes[:-1], changes[1:]):
                        deg_val = int(nd_sorted[a])
                        buckets[deg_val].extend(nb_sorted[a:b].tolist())

            r += 1
        return

    def compute_high_low(self):
        """Compute neighborhood lifetime relations."""
        n = self.len
        self.nbs_high = np.empty(n, dtype=np.int32)
        self.nbs_low = np.empty(n, dtype=np.int32)
        lt = np.asarray(self.lifetime, dtype=np.intp)
        indptr, indices = self.indptr, self.indices

        for i in range(n):
            s, e = indptr[i], indptr[i + 1]
            if s == e:
                self.nbs_high[i] = 0
                self.nbs_low[i] = 0
                continue
            li = lt[i]
            nb_lt = lt[indices[s:e]]
            self.nbs_high[i] = np.count_nonzero(li < nb_lt)
            self.nbs_low[i] = np.count_nonzero(li > nb_lt)

    def _csr_from_indptr_indices(self):
        """Construct sparse (CSR) matrix with ones on edges."""
        n = self.len
        data = np.ones(self.indices.shape[0], dtype=np.uint8)
        A = csr_matrix((data, self.indices, self.indptr), shape=(n, n))
        return A

    def cluster(self):
        """Lifetime clustering."""
        self.compute_high_low()

        labels = self.labels_
        indptr, indices = self.indptr, self.indices
        start_deg = self.start_degree
        nbs_high, nbs_low = self.nbs_high, self.nbs_low

        noise = start_deg == 0
        inner = (nbs_high < nbs_low) & (~noise)
        seeds = (nbs_high == 0) & (nbs_low > 0) & (~noise)

        labels[noise] = -1

        # inner point assignment
        if not np.any(inner):
            labels[(labels == -3)] = -2
        else:
            A = self._csr_from_indptr_indices()
            A_inner = A[inner][:, inner]

            n_comp, comp_ids = connected_components(A_inner, directed=False)

            inner_idx = np.flatnonzero(inner)
            seeds_idx = np.flatnonzero(seeds)

            comp_has_seed = np.zeros(n_comp, dtype=bool)
            if seeds_idx.size:
                is_seed_local = np.zeros(inner_idx.size, dtype=bool)
                seed_set = set(seeds_idx.tolist())
                for i, g in enumerate(inner_idx):
                    if g in seed_set:
                        is_seed_local[i] = True
                np.logical_or.at(comp_has_seed, comp_ids, is_seed_local)

            comps = []
            if seeds_idx.size:
                comps = [(0, comp) for comp in range(n_comp) if comp_has_seed[comp]]

            labels[(labels == -3)] = -2

            cluster_id = 0
            for _, comp in comps:
                loc = np.flatnonzero(comp_ids == comp)
                glb = inner_idx[loc]
                labels[glb] = cluster_id
                cluster_id += 1

        # outer point assignment
        for idx in self.sorted_ids:
            if labels[idx] != -2:
                continue
            s, e = indptr[idx], indptr[idx + 1]
            nb_labels = labels[indices[s:e]]
            valid = nb_labels[nb_labels >= 0]
            if valid.size == 0:
                labels[idx] = -1
                continue
            counts = np.bincount(valid)
            labels[idx] = counts.argmax()

    def fit(self, data):
        """Compute clusters from a data matrix and predict labels.

        Parameters
        ----------
        data: array-like of shape (n_samples, n_features)
            The data points to cluster.
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.len = len(self.data)
        self.labels_ = np.full(self.len, -3, dtype=np.int32)

        self.index_points()
        self.contraction()
        self.cluster()
