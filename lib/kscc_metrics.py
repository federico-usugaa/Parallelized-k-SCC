"""Unsupervised evaluation metrics."""

# Authors: Robert Layton <robertlayton@gmail.com>
#          Arnaud Fouchet <foucheta@gmail.com>
#          Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn.preprocessing import LabelEncoder


def check_number_of_labels(n_labels, n_samples):
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
                         "to n_samples - 1 (inclusive)" % n_labels)


def silhouette_score(X, labels, _distance):
    silhouetteResult = silhouette_samples(X, labels, _distance)

    return silhouetteResult, np.mean(silhouetteResult)


def silhouette_samples(X, labels, _distances):
    """Compute the Silhouette Coefficient for each sample.

    The Silhouette Coefficient is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    Silhouette Coefficient are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.
    Note that Silhouette Coefficient is only defined if number of labels
    is 2 <= n_labels <= n_samples - 1.

    This function returns the Silhouette Coefficient for each sample.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.

    Read more in the :ref:`User Guide <silhouette_coefficient>`.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.

    labels : array, shape = [n_samples]
             label values for each sample

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`. If X is
        the distance array itself, use "precomputed" as the metric.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    silhouette : array, shape = [n_samples]
        Silhouette Coefficient for each samples.

    References
    ----------

    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_

    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    """
    # X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    check_number_of_labels(len(le.classes_), X.shape[0])

    distances = _distances
    unique_labels = le.classes_
    n_samples_per_label = np.bincount(labels, minlength=len(unique_labels))

    # For sample i, store the mean distance of the cluster to which
    # it belongs in intra_clust_dists[i]
    intra_clust_dists = np.zeros(distances.shape[0], dtype=distances.dtype)

    # For sample i, store the mean distance of the second closest
    # cluster in inter_clust_dists[i]
    inter_clust_dists = np.inf + intra_clust_dists

    for curr_label in range(len(unique_labels)):

        # Find inter_clust_dist for all samples belonging to the same
        # label.
        mask = labels == curr_label
        current_distances = distances[mask]

        # Leave out current sample.
        n_samples_curr_lab = n_samples_per_label[curr_label] - 1
        if n_samples_curr_lab != 0:
            intra_clust_dists[mask] = np.sum(
                current_distances[:, mask], axis=1) / n_samples_curr_lab

        # Now iterate over all other labels, finding the mean
        # cluster distance that is closest to every sample.
        for other_label in range(len(unique_labels)):
            if other_label != curr_label:
                other_mask = labels == other_label
                other_distances = np.mean(
                    current_distances[:, other_mask], axis=1)
                inter_clust_dists[mask] = np.minimum(
                    inter_clust_dists[mask], other_distances)

    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # score 0 for clusters of size 1, according to the paper
    sil_samples[n_samples_per_label.take(labels) == 1] = 0
    return sil_samples