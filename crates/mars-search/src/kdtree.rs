//! A bucketed k-d tree with approximate nearest-neighbour search, ported from
//! `reference/mars1/nn_search.c` (Matthias Ruhl, GPL) for the Saupe family
//! (`Saupe`, `Saupe-Fisher`, `Mc-Saupe`).
//!
//! Ported faithfully rather than replaced with a generic nearest-neighbour crate,
//! per the Step 9 brief: max-spread-dimension split at the median, the
//! best-bounding-box-first priority traversal, and the `eps`-approximate early
//! termination (`found < num || eps2*best_remaining_bbox_dist < dist2[num-1]`).
//!
//! **Determinism, not bit-exactness.** `nn_search.c`'s own `compare()` returns `1` (never
//! `0`) for equal keys, which under `qsort`'s *unspecified* order-among-equals makes the
//! reference's own left/right split among tied points implementation-defined — the Step 9
//! brief explicitly says search behaviour need not reproduce this bit-for-bit any more.
//! This port uses a valid comparator and stable sort, retaining input order for tied
//! coordinates. Unlike the C comparator, this also satisfies Rust's sorting contract
//! for constant blocks and other repeated feature vectors.

const BUCKETSIZE: usize = 10;

enum KdNodeKind {
    Leaf(Vec<usize>),
    Split {
        cutdim: usize,
        cutval: f32,
        left: Box<KdNode>,
        right: Box<KdNode>,
    },
}

/// One k-d tree node. `min`/`max` are the node's bounding box over all `dim` feature axes
/// — kept on every node (leaf and internal alike), exactly as `nn_search.c` does, since
/// `kdtree_search`'s heap prioritises nodes by distance to this box regardless of node
/// kind.
pub struct KdNode {
    min: Vec<f32>,
    max: Vec<f32>,
    kind: KdNodeKind,
}

/// Build a k-d tree over `points` (each an f32 feature vector of length `dim`).
/// `kdtree_build`. Returns `None` for an empty point set (nothing to search).
pub fn build(points: &[Vec<f32>], dim: usize) -> Option<KdNode> {
    if points.is_empty() {
        return None;
    }
    let idx: Vec<usize> = (0..points.len()).collect();
    Some(build_rec(points, dim, idx))
}

fn build_rec(points: &[Vec<f32>], dim: usize, mut idx: Vec<usize>) -> KdNode {
    let num = idx.len();
    let mut min = vec![0.0f32; dim];
    let mut max = vec![0.0f32; dim];
    let mut cutdim = 0usize;
    let mut spread = -1.0f32;
    for d in 0..dim {
        let mut mn = points[idx[0]][d];
        let mut mx = mn;
        for &i in &idx[1..] {
            let v = points[i][d];
            if mn > v {
                mn = v;
            } else if mx < v {
                mx = v;
            }
        }
        if mx - mn > spread {
            spread = mx - mn;
            cutdim = d;
        }
        min[d] = mn;
        max[d] = mx;
    }

    if num <= BUCKETSIZE {
        return KdNode {
            min,
            max,
            kind: KdNodeKind::Leaf(idx),
        };
    }

    idx.sort_by(|&a, &b| {
        points[a][cutdim]
            .partial_cmp(&points[b][cutdim])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let cutval = (points[idx[num / 2 - 1]][cutdim] + points[idx[num / 2]][cutdim]) / 2.0;
    let right_idx = idx.split_off(num / 2);
    let left = build_rec(points, dim, idx);
    let right = build_rec(points, dim, right_idx);
    KdNode {
        min,
        max,
        kind: KdNodeKind::Split {
            cutdim,
            cutval,
            left: Box::new(left),
            right: Box::new(right),
        },
    }
}

/// Squared distance from `q` to a node's bounding box (`hpush`'s inline computation) —
/// zero if `q` is inside the box on every axis.
fn bbox_dist2(node: &KdNode, q: &[f32]) -> f32 {
    let mut d = 0.0f32;
    for (i, &qi) in q.iter().enumerate() {
        if qi < node.min[i] {
            let t = node.min[i] - qi;
            d += t * t;
        } else if qi > node.max[i] {
            let t = qi - node.max[i];
            d += t * t;
        }
    }
    d
}

/// `kdtree_search`: up to `num` approximate `(1+eps)`-nearest neighbours of `q`, as
/// indices into the `points` array the tree was built from. Returns them nearest-first.
pub fn search(q: &[f32], points: &[Vec<f32>], tree: &KdNode, eps: f32, num: usize) -> Vec<usize> {
    // A binary min-heap of (bbox_dist2, node), exactly `hpush`/`hpop`'s role — priority
    // is "closest possible point could be in this box", so unexplored regions that can't
    // beat the current worst kept candidate are never opened.
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<HeapItem<'_>>> =
        std::collections::BinaryHeap::new();
    heap.push(std::cmp::Reverse(HeapItem {
        dist: bbox_dist2(tree, q),
        node: tree,
    }));

    let mut nlist: Vec<usize> = Vec::with_capacity(num);
    let mut dist2: Vec<f32> = Vec::with_capacity(num);
    let eps2 = (1.0 + eps) * (1.0 + eps);

    while let Some(std::cmp::Reverse(HeapItem { dist: top_dist, .. })) = heap.peek() {
        let stop = nlist.len() >= num && eps2 * top_dist >= *dist2.last().unwrap();
        if stop {
            break;
        }
        let std::cmp::Reverse(HeapItem { node, .. }) = heap.pop().unwrap();

        let mut cur = node;
        loop {
            match &cur.kind {
                KdNodeKind::Split {
                    cutdim,
                    cutval,
                    left,
                    right,
                } => {
                    if q[*cutdim] < *cutval {
                        heap.push(std::cmp::Reverse(HeapItem {
                            dist: bbox_dist2(right, q),
                            node: right,
                        }));
                        cur = left;
                    } else {
                        heap.push(std::cmp::Reverse(HeapItem {
                            dist: bbox_dist2(left, q),
                            node: left,
                        }));
                        cur = right;
                    }
                }
                KdNodeKind::Leaf(members) => {
                    for &pi in members {
                        let mut d = 0.0f32;
                        for k in 0..q.len() {
                            let t = q[k] - points[pi][k];
                            d += t * t;
                        }
                        if nlist.len() == num && d > *dist2.last().unwrap() {
                            continue;
                        }
                        if nlist.len() < num {
                            nlist.push(0);
                            dist2.push(0.0);
                        }
                        let mut j = nlist.len() - 1;
                        while j > 0 && dist2[j - 1] > d {
                            dist2[j] = dist2[j - 1];
                            nlist[j] = nlist[j - 1];
                            j -= 1;
                        }
                        dist2[j] = d;
                        nlist[j] = pi;
                    }
                    break;
                }
            }
        }
    }

    nlist
}

struct HeapItem<'a> {
    dist: f32,
    node: &'a KdNode,
}
impl PartialEq for HeapItem<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for HeapItem<'_> {}
impl PartialOrd for HeapItem<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(rows: &[[f32; 2]]) -> Vec<Vec<f32>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    #[test]
    fn finds_exact_nearest_neighbour_on_a_small_grid() {
        let points = pts(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [10.0, 10.0]]);
        let tree = build(&points, 2).unwrap();
        let found = search(&[0.1, 0.1], &points, &tree, 0.0, 1);
        assert_eq!(found, vec![0]);
    }

    #[test]
    fn returns_k_nearest_in_order_with_more_than_bucketsize_points() {
        // 30 points on a line: 0.0, 1.0, .., 29.0 -- forces an internal split.
        let points: Vec<Vec<f32>> = (0..30).map(|i| vec![i as f32]).collect();
        let tree = build(&points, 1).unwrap();
        let found = search(&[14.6], &points, &tree, 0.0, 3);
        assert_eq!(found, vec![15, 14, 16]);
    }

    #[test]
    fn empty_point_set_builds_no_tree() {
        let points: Vec<Vec<f32>> = vec![];
        assert!(build(&points, 2).is_none());
    }
}
