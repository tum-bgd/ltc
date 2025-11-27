//! # Lifetime Clustering (LTC)
//!
//! Reference: J M Zollner, B Teuscher, W Mansour, and M Werner, "Efficent and robust topology-based clustering"

use std::collections::VecDeque;

use num_traits::Float;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use srtree::{Euclidean, SRTree};

/// Label values.
pub type Labels = Vec<i32>;
/// Lifetime values.
pub type Lifetimes = Vec<i32>;

/// Fit model.
pub fn fit<T>(data: &[Vec<T>], eps: T) -> (Labels, Lifetimes)
where
    T: Float + Send + Sync,
{
    let indices = build_neighbour_graph(data, eps);

    let (labels, lifetime) = fit_with(&indices);

    (labels, lifetime)
}

/// Fit model from neighbourhood indices.
pub fn fit_with(indices: &[Vec<usize>]) -> (Labels, Lifetimes) {
    let (lifetimes, sorted_ids) = compute_lifetime(indices);

    let labels = cluster(indices, &lifetimes, &sorted_ids);

    (labels, lifetimes)
}

/// Construct fixed radius neighbour graph.
fn build_neighbour_graph<T>(data: &[Vec<T>], eps: T) -> Vec<Vec<usize>>
where
    T: Float + Send + Sync,
{
    let index = SRTree::default(data, Euclidean::default()).unwrap();

    data.par_iter()
        .enumerate()
        .map(|(row_i, p)| {
            // radius search
            let mut neighbours = index.query_radius(p, eps);

            // drop diagonal i.e. self edges
            neighbours.retain(|col_id| *col_id != row_i);

            neighbours
        })
        .collect()
}

/// Compute lifetimes.
fn compute_lifetime(indices: &[Vec<usize>]) -> (Lifetimes, Vec<usize>) {
    let n = indices.len();

    let mut lifetimes = vec![-1; n];
    let mut sorted_ids = vec![0; n];
    let mut filter = vec![false; n];

    // buckets[d] has nodes with degree d
    let mut buckets = Vec::with_capacity(8);
    let mut degree: Vec<_> = indices
        .iter()
        .enumerate()
        .map(|(idx, neighbours)| {
            let d = neighbours.len();

            while d >= buckets.len() {
                buckets.push(Vec::with_capacity(n / 8));
            }
            unsafe { buckets.get_unchecked_mut(d).push(idx) };

            d
        })
        .collect();

    let mut rpos = n; // pointer for reverse removal order
    let mut r = 1; // round
    let mut current_min_deg = 0;

    let mut worklist: Vec<usize> = Vec::with_capacity(64);

    while current_min_deg < buckets.len() {
        let mut current_bucket = std::mem::take(&mut buckets[current_min_deg]);

        // filter points without lifetime
        current_bucket.retain(|idx| *unsafe { lifetimes.get_unchecked(*idx) } == -1);

        // skip empty degree bucket
        if current_bucket.is_empty() {
            current_min_deg += 1;
            continue;
        }

        for idx in current_bucket {
            // assign lifetime
            *unsafe { lifetimes.get_unchecked_mut(idx) } = r;

            // record reverse removal order
            rpos -= 1;
            *unsafe { sorted_ids.get_unchecked_mut(rpos) } = idx;

            // flag neighbors once (no duplicates)
            for &nb in unsafe { indices.get_unchecked(idx) }.iter() {
                if unsafe { *lifetimes.get_unchecked(nb) } != -1 {
                    continue; // already removed
                }

                let is_known = unsafe { filter.get_unchecked_mut(nb) };
                if !*is_known {
                    *is_known = true;
                    worklist.push(nb);
                }
            }
        }

        // each neighbor is decremented only once per round
        for nb in worklist.drain(..) {
            // reset flag
            unsafe { *filter.get_unchecked_mut(nb) = false };

            // skip points with assigned lifetime
            if *unsafe { lifetimes.get_unchecked(nb) } != -1 {
                continue;
            }

            let deg = unsafe { degree.get_unchecked_mut(nb) };

            // skip points that can not appear in uppcoming buckets
            if *deg < current_min_deg {
                continue;
            };

            // update degree
            *deg -= 1;

            // add point to buckets
            unsafe { buckets.get_unchecked_mut(*deg).push(nb) };
        }

        r += 1
    }

    (lifetimes, sorted_ids)
}

/// Assign labels.
fn cluster(indices: &[Vec<usize>], lifetimes: &[i32], sorted_ids: &[usize]) -> Labels {
    let n = indices.len();
    assert_eq!(lifetimes.len(), n);
    assert_eq!(sorted_ids.len(), n);

    let mut labels = vec![-2; n];
    let mut filter = vec![false; n];
    let mut seeds_idx = Vec::with_capacity(32);

    for (idx, lt) in lifetimes.iter().enumerate() {
        // compute neighbour lifetime relations
        let (mut low, mut high) = (0, 0);

        let indices = unsafe { indices.get_unchecked(idx) };

        for ii in indices.iter() {
            let nb_lt = unsafe { lifetimes.get_unchecked(*ii) };
            if lt < nb_lt {
                high += 1;
            }
            if lt > nb_lt {
                low += 1;
            }
        }

        // asigne label
        let label = unsafe { labels.get_unchecked_mut(idx) };

        if indices.is_empty() {
            // noise
            *label = -1;
        } else {
            if high < low {
                // inner
                *unsafe { filter.get_unchecked_mut(idx) } = true;
            }
            if high == 0 && low > 0 {
                // seed (subset of inner)
                seeds_idx.push(idx);
            }
        }
    }

    // Inner point assignment
    let mut cluster_id = 0;
    let mut to_visit = VecDeque::with_capacity(64);

    for seed_idx in seeds_idx {
        // skip already visited seeds
        if !unsafe { *filter.get_unchecked(seed_idx) } {
            continue;
        }

        to_visit.push_back(seed_idx);

        while let Some(idx) = to_visit.pop_front() {
            let first_visit = unsafe { filter.get_unchecked_mut(idx) };
            // only visit once
            if *first_visit {
                // mark as visied
                *first_visit = false;

                // set label
                *unsafe { labels.get_unchecked_mut(idx) } = cluster_id;

                // traverse
                for nb in unsafe { indices.get_unchecked(idx).iter() } {
                    // only follow unvisited edges between inner points
                    if *unsafe { filter.get_unchecked(*nb) } {
                        to_visit.push_back(*nb);
                    }
                }
            }
        }

        cluster_id += 1;
    }

    // Outer point assignment.
    let mut counts: Vec<usize> = vec![0; cluster_id as usize + 1];

    for &idx in sorted_ids.iter() {
        // skip already labeled points
        if unsafe { *labels.get_unchecked(idx) } != -2 {
            continue;
        }

        // count labeled neighbors (sparse over labels seen)
        counts.fill_with(Default::default);

        // pick argmax (tie break by first seen)
        let mut max_count = 0;
        let mut max_label = -1;

        for &nb in unsafe { indices.get_unchecked(idx) }.iter() {
            let nb_label = unsafe { *labels.get_unchecked(nb) };

            // ignore noise/unlabeled
            if nb_label < 0 {
                continue;
            }

            // increment
            let count = unsafe { counts.get_unchecked_mut(nb_label as usize) };
            *count += 1;

            // update max
            if *count > max_count {
                max_count = *count;
                max_label = nb_label;
            }
        }

        *unsafe { labels.get_unchecked_mut(idx) } = max_label;
    }

    labels
}
