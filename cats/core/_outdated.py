import numpy as np
from scipy import optimize
from scipy.spatial.distance import cdist


_metrics = {'cityblock': 1, 'euclidean': 2,
            1: 'cityblock', 2: 'euclidean'}


def CheckDistance(vec1, vec2, max_dist, metric):
    assert vec1.shape[-1] == vec2.shape[-1]
    dim = vec1.shape[-1]
    max_dist = max_dist if (max_dist is not None) else np.inf
    metric = metric if (metric is not None) else 1

    if isinstance(max_dist, (tuple, list, np.ndarray)):
        max_dist = max_dist + type(max_dist)([np.inf] * (dim - len(max_dist)))

        R = np.abs(np.expand_dims(vec1, 1) - np.expand_dims(vec2, 0))
        bool_inds = np.prod(R <= np.expand_dims(max_dist, (0, 1)), axis=-1, dtype=bool)
        norm = _metrics.get(metric, 1) if isinstance(metric, str) else metric
        R = np.linalg.norm(R, ord=norm, axis=-1)

    elif isinstance(max_dist, (int, float)):
        norm = _metrics.get(metric, 'cityblock') if isinstance(metric, int) else metric
        R = cdist(vec1, vec2, metric=norm)
        bool_inds = (R <= max_dist)
    else:
        raise ValueError(f"`max_dist` must be `int`/`float` or `array[`int`/`float`]`, given {type(max_dist)}")
    R[~bool_inds] = float('inf')
    return R, bool_inds


def SplitByDistance(vec1, vec2, max_dist, metric):
    R, bool_inds = CheckDistance(vec1, vec2, max_dist, metric)

    close_sums = tuple(bool_inds.sum(axis=dim) for dim in (1, 0))
    distant_inds = tuple(cs == 0 for cs in close_sums)

    paired_inds = tuple(cs == 1 for cs in close_sums)
    pairs = np.argwhere(bool_inds * paired_inds[0][:, None] * paired_inds[1][None, :])
    pairs = tuple(pairs.T)

    close_sums[0][pairs[0]] = -1
    close_sums[1][pairs[1]] = -1
    close_inds = tuple(cs > 0 for cs in close_sums)

    R = R[close_inds[0]][:, close_inds[1]]

    return R, distant_inds, close_inds, pairs


def LSAMatchPair(sequence1, sequence2, max_dist=None, metric=1):
    seq1, seq2 = sequence1, sequence2

    R, *inds = SplitByDistance(seq1, seq2, max_dist, metric)
    matched_1, matched_2 = optimize.linear_sum_assignment(R)
    del R  # to clean memory

    distant_inds_1, distant_inds_2 = inds[0]
    close_inds_1, close_inds_2 = inds[1]
    pairs_1, pairs_2 = inds[2]

    inds1, inds2 = np.arange(len(seq1)), np.arange(len(seq2))
    reindexed_1, reindexed_2 = inds1[close_inds_1], inds2[close_inds_2]

    n_distant_1 = np.count_nonzero(distant_inds_1)
    n_distant_2 = np.count_nonzero(distant_inds_2)

    # Elements of sequences may not always have a pair
    remained_1 = list(set(range(len(reindexed_1))) - set(matched_1))
    remained_2 = list(set(range(len(reindexed_2))) - set(matched_2))

    nan_inds = lambda x: np.full(x, -1)

    matched_seq1_inds = (pairs_1,
                         reindexed_1[matched_1],
                         reindexed_1[remained_1],
                         nan_inds(len(remained_2)),
                         inds1[distant_inds_1],
                         nan_inds(n_distant_2))
    matched_seq2_inds = (pairs_2,
                         reindexed_2[matched_2],
                         nan_inds(len(remained_1)),
                         reindexed_2[remained_2],
                         nan_inds(n_distant_1),
                         inds2[distant_inds_2])
    matched = (np.concatenate(matched_seq1_inds),
               np.concatenate(matched_seq2_inds))
    return matched


def give_unpaired(n1, n2, pairs):
    i1, i2 = tuple(set(pi) for pi in np.array(pairs).reshape(-1, 2).T)
    ref1, ref2 = set(range(n1)), set(range(n2))
    ind1, ind2 = list(ref1 - i1), list(ref2 - i2)
    return ind1, ind2


def sparsify_path(path, vec1, vec2, dist):
    N = len(path)
    sparse, buffer = [], []
    i0, j0 = -1, -1
    for cnt, (i, j) in enumerate(path):
        dist_ij = dist(vec1[i], vec2[j])

        if ((i != i0) and (j != j0) and len(buffer) > 0) or (last := (cnt == N - 1)):
            if last:
                buffer.append(((i, j), dist_ij))
            min_ind, min_dist = min(buffer, key=lambda x: x[-1])
            if np.isfinite(min_dist):
                sparse.append(min_ind)
            buffer.clear()
        buffer.append(((i, j), dist_ij))
        i0, j0 = i, j

    ind1, ind2 = give_unpaired(len(vec1), len(vec2), sparse)
    return sparse, (ind1, ind2)