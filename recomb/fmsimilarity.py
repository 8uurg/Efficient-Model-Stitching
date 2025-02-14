#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_pre_pipeline():
    return make_pipeline(StandardScaler())


# -- PCA-based Similarity --
from sklearn.decomposition import PCA


def compute_pca_feature_map_similarity(
    X_a, X_b, nc, nc_a=None, nc_b=None, whiten=True, normalize=True
):
    # nc = 1
    if nc_a is None:
        nc_a = nc
    if nc_b is None:
        nc_b = nc

    X_ab = np.concatenate([X_a, X_b], axis=1)

    pre_a = get_pre_pipeline()
    pre_b = get_pre_pipeline()
    pre_ab = get_pre_pipeline()

    if normalize:
        X_a = pre_a.fit_transform(X_a)
        X_b = pre_b.fit_transform(X_b)
        X_ab = pre_ab.fit_transform(X_ab)

    # nc_a = app_state["0_f"].shape[1] // 2
    # nc_b = app_state["1_f"].shape[1] // 2

    def get_pca_pipeline(n_components):
        return make_pipeline(PCA(n_components=n_components, whiten=whiten))

    try:
        # Limit to the number of samples & features.
        if nc_a >= 1.0:
            nc_a = min(nc_a, X_a.shape[0], X_a.shape[1])

        if nc_b >= 1.0:
            nc_b = min(nc_b, X_b.shape[0], X_b.shape[1])
    except:
        pass

    pca_a = get_pca_pipeline(nc_a)
    pca_b = get_pca_pipeline(nc_b)
    # pca_ab_less = get_pca_pipeline(max(nc_a, nc_b))

    pca_a.fit(X_a)
    pca_b.fit(X_b)
    # pca_ab_less.fit(X_ab)

    Xt_a = pca_a.transform(X_a)
    Xt_b = pca_b.transform(X_b)
    Xp_a = pca_a.inverse_transform(Xt_a)
    Xp_b = pca_b.inverse_transform(Xt_b)

    # Xt_a.shape[0]
    pca_ab = get_pca_pipeline(min(Xt_a.shape[0], Xt_a.shape[1] + Xt_b.shape[1]))
    pca_ab.fit(X_ab)
    Xp_ab = pca_ab.inverse_transform(pca_ab.transform(X_ab))
    # Xp_ab_less = (pca_ab_less.inverse_transform(pca_ab_less.transform(X_ab)))

    err_a = ((Xp_a - X_a) ** 2).sum()
    err_b = ((Xp_b - X_b) ** 2).sum()
    # err_ab_less = ((Xp_ab_less - X_ab)**2).sum()
    err_ab = ((Xp_ab - X_ab) ** 2).sum()

    # The reconstruction error of the combined network should be higher than the individual networks
    # given the same number of components, as it has to reconstruct more.
    # If the two networks are completely independent, their errors should sum.
    # If we have the same network twice, we are completely redundant - and the error should be
    # equal for the same-n-components model and lower for the multiple components model.
    # i.e.
    # err_ab <= err_a + err_b
    # as err_a and err_b would be the errors if accomplishing this independently.

    eps = 1e-5
    err_ratio = err_ab / (err_a + err_b + eps)
    # print(f"individual errors; a: {err_a} b: {err_b}. Combined; less: {err_ab_less}; more: {err_ab}. {} ")
    return min(1.0, max(0.0, err_ratio))


# -- Regression Based Similarity --
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor


def compute_regressor_feature_map_similarity(
    X_a, X_b, normalize=True, BaseRegressor=LinearRegression
):
    pre_a = get_pre_pipeline()
    pre_b = get_pre_pipeline()

    if normalize:
        X_a = pre_a.fit_transform(X_a)
        X_b = pre_b.fit_transform(X_b)

    reg_a_to_b = MultiOutputRegressor(BaseRegressor())
    reg_b_to_a = MultiOutputRegressor(BaseRegressor())

    reg_a_to_b.fit(X_a, X_b)
    reg_b_to_a.fit(X_b, X_a)

    # we will assume that score >= 0.0, as this is a regressor trained to perform this task.
    # by sklearn documentation, score <= 1.0
    sab = max(0.0, reg_a_to_b.score(X_a, X_b))
    sba = max(0.0, reg_b_to_a.score(X_b, X_a))
    return 1.0 - (sab + sba) / 2


# -- Simply-do-nothing Similarity --
def compute_mock_similarity(X_a, X_b):
    return 0.0

# -- Linear CKA --
def compute_linear_cka_feature_map_similarity(
    X_a, X_b, align="pad-zero", subsample=True
):
    # max_samples = 256
    max_samples = 16

    assert X_a.shape[0] == X_b.shape[0]
    if X_a.shape[0] > max_samples:
        # Too many samples make the gram matrix too large.
        # Limit the number of samples so that this is not an issue.
        if subsample:
            X_a = X_a[:max_samples, :]
            X_b = X_b[:max_samples, :]
        else:
            raise ValueError("too many samples: matrix will be too big")

    if X_a.shape[1] != X_b.shape[1]:
        if align == "pad-zero":
            nf = max(X_a.shape[1], X_b.shape[1])
            X_a = np.pad(X_a, [(0, 0), (0, nf - X_a.shape[1])])
            X_b = np.pad(X_b, [(0, 0), (0, nf - X_b.shape[1])])
        else:
            raise ValueError("Mismatched Shapes")

    # Center X_a and X_b
    X_a = X_a - X_a.mean(axis=0, keepdims=True)
    X_b = X_b - X_b.mean(axis=0, keepdims=True)

    gram_X_a = X_a.dot(X_a.T)
    gram_X_b = X_b.dot(X_b.T)

    numerator = gram_X_a.ravel().dot(gram_X_b.ravel())
    denominator = np.linalg.norm(gram_X_a) * np.linalg.norm(gram_X_b)
    eps = 0
    return 1 - (numerator / (denominator + eps))


# Ordered Matching
import igraph as ig
from .pqueue import KeyedPriorityQueue


def dag_max_distance_from_roots(g: ig.Graph):
    ord = g.topological_sorting(mode="out")
    g.vs["o"] = 0
    for i in ord:
        v = g.vs[i]
        for n in v.neighbors(mode="out"):
            n["o"] = max(n["o"], v["o"] + 1)


def hashset(v):
    return sum(2 >> x for x in v)


def find_ordered_matching(
    S: np.ndarray, graph_a: ig.Graph, graph_b: ig.Graph, verbose=False
):
    """
    Given a pairwise similarity matrix for two graphs, a & b, find a matching between the two networks
    va_i <-> v_b_j, such that a graph that merges these two nodes preserves the property of being a
    directed acyclic graph (DAG) while maximizing similarity.
    """

    # Make copies, to be sure
    g_a = graph_a.copy()
    g_b = graph_b.copy()

    # Derive maximum distance from root - as to ensure ordering is valid.
    dag_max_distance_from_roots(g_a)
    dag_max_distance_from_roots(g_b)

    # Sidenote: It might be better to use a particular heuristic and apply A*...
    #
    def determine_priority(vs_a, vs_b, value):
        # Priority, we want to visit states which could potentially visit the other states first.
        os_a = g_a.vs[vs_a]["o"]
        os_b = g_b.vs[vs_b]["o"]
        os = os_a + os_b
        os.sort()
        return os

    # Start off with all the input nodes.
    inputs_a = [v.index for v in g_a.vs if v.indegree() == 0]
    inputs_b = [v.index for v in g_b.vs if v.indegree() == 0]
    initial = (frozenset(inputs_a), frozenset(inputs_b))
    v: Dict[Tuple[frozenset[int], frozenset[int]], Any] = {initial: (0.0, {})}

    # Set up priority queue with initial point, and store the corresponding key, for completeness
    q = KeyedPriorityQueue([(determine_priority(inputs_a, inputs_b, 0), initial)])
    # m = {initial: 0}

    def update(state, value):
        if state not in v:
            k = q.add(determine_priority(state[0], state[1], value), state)
            # m[state] = k
            v[state] = value
            if verbose:
                print(f"new state: {state} -> {value}")
        else:
            current_value = v.get(state)
            if current_value[0] < value[0]:
                v[state] = value
                if verbose:
                    print(f"improved state: {state} -> {value}, was {current_value}")

    best_s = None

    while not q.is_empty():
        _, _, state = q.popmin()
        if verbose:
            print(f"visiting state {state}")
        # del m[v[1]]
        s = v[state]
        del v[state]
        if len(state[0]) == 0 or len(state[1]) == 0:
            if best_s is None or best_s[0] < s[0]:
                best_s = s

            # Skip over the remainder of this function - as one of the sets is empty
            # there are no further possible matches!
            continue

        v_a, v_b = state

        # We can either proceed (skip) for any over the elements in a, not increasing similarity or adding matchings.
        for a in v_a:
            vnew_a = frozenset(g_a.neighbors(a, mode="out")).union(v_a) - frozenset(
                g_a.subcomponent(a, mode="in")
            )
            for x in vnew_a:
                assert (
                    x < S.shape[0]
                ), f"tried adding node for graph index {x} but matrix is of size {S.shape[0]}"
            update((vnew_a, v_b), s)

        # Similarly, for b.
        for b in v_b:
            vnew_b = frozenset(g_b.neighbors(b, mode="out")).union(v_b) - frozenset(
                g_b.subcomponent(b, mode="in")
            )
            for x in vnew_b:
                assert (
                    x < S.shape[1]
                ), f"tried adding node for graph index {x} but matrix is of size {S.shape[1]}"
            update((v_a, vnew_b), s)

        # Alternatively, we can match any of the two in a and b.
        for a in v_a:
            vnew_a = frozenset(g_a.neighbors(a, mode="out")).union(v_a) - frozenset(
                g_a.subcomponent(a, mode="in")
            )
            for x in vnew_a:
                assert (
                    x < S.shape[0]
                ), f"tried adding node for graph index {x} but matrix is of size {S.shape[0]}"
            for b in v_b:
                vnew_b = frozenset(g_b.neighbors(b, mode="out")).union(v_b) - frozenset(
                    g_b.subcomponent(b, mode="in")
                )
                for x in vnew_b:
                    assert (
                        x < S.shape[1]
                    ), f"tried adding node for graph index {x} but matrix is of size {S.shape[1]}"
                d = {a: b}
                d.update(s[1])
                update((vnew_a, vnew_b), (s[0] + S[a, b], d))

    return best_s


def test_find_ordered_matching_0():
    g_a = ig.Graph(
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
        directed=True,
    )
    g_b = ig.Graph(
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
        directed=True,
    )

    S = np.zeros(shape=(len(g_a.vs), len(g_b.vs)))
    S[0, 0] = 1
    S[1, 2] = 1
    S[2, 1] = 1
    S[3, 3] = 1
    S[4, 4] = 1
    S[5, 5] = 1

    r = find_ordered_matching(S, g_a, g_b)
    assert np.isclose(r[0], 6.0)
    assert r[1][0] == 0
    assert r[1][1] == 2
    assert r[1][2] == 1
    assert r[1][3] == 3
    assert r[1][4] == 4
    assert r[1][5] == 5


def test_find_ordered_matching_1():
    g_a = ig.Graph(
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
        directed=True,
    )
    g_b = ig.Graph(
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
        directed=True,
    )

    S = np.zeros(shape=(len(g_a.vs), len(g_b.vs)))
    S[0, 0] = 1
    S[1, 1] = 1
    S[2, 2] = 1
    S[3, 3] = 1
    S[4, 4] = 1
    S[5, 5] = 1

    r = find_ordered_matching(S, g_a, g_b)
    assert np.isclose(r[0], 6.0)
    assert r[1][0] == 0
    assert r[1][1] == 1
    assert r[1][2] == 2
    assert r[1][3] == 3
    assert r[1][4] == 4
    assert r[1][5] == 5


def test_find_ordered_matching_2():
    g_a = ig.Graph(
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
        directed=True,
    )
    g_b = ig.Graph(
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
        directed=True,
    )

    S = np.zeros(shape=(len(g_a.vs), len(g_b.vs)))
    S[0, 0] = 1
    S[1, 3] = 1
    S[3, 1] = 1

    r = find_ordered_matching(S, g_a, g_b)
    assert np.isclose(r[0], 2.0)
    assert r[1][0] == 0


def test_find_ordered_matching_3():
    g_a = ig.Graph(
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
        directed=True,
    )
    g_b = ig.Graph(
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
        directed=True,
    )

    S = np.zeros(shape=(len(g_a.vs), len(g_b.vs)))
    S[0, 0] = 1
    S[1, 3] = 1
    S[2, 2] = 1
    S[3, 1] = 1
    S[4, 4] = 1
    S[5, 5] = 1

    r = find_ordered_matching(S, g_a, g_b)
    assert np.isclose(r[0], 5.0)
    assert r[1][0] == 0
    # assert r[1][1] == 3
    assert r[1][2] == 2
    # assert r[1][3] == 1
    assert r[1][4] == 4
    assert r[1][5] == 5


def test_find_ordered_matching_4():
    g_a = ig.Graph(
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
        directed=True,
    )
    g_b = ig.Graph(
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
        directed=True,
    )

    S = np.zeros(shape=(len(g_a.vs), len(g_b.vs)))
    S[0, 0] = 1
    S[1, 4] = 1
    S[2, 2] = 1
    S[3, 3] = 1
    S[4, 1] = 1
    S[5, 5] = 1

    r = find_ordered_matching(S, g_a, g_b)
    assert np.isclose(r[0], 6.0)
    assert r[1][0] == 0
    assert r[1][1] == 4
    assert r[1][2] == 2
    assert r[1][3] == 3
    assert r[1][4] == 1
    assert r[1][5] == 5
