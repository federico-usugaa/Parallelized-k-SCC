#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Noted by Tai Dinh
This file is used to run the Silhouette based algorithm for categorical data clustering (k-SCC algorithm),
using the Kernel density estimation approach and information theoretic based dissimilarity measure, global_attr_freq = 1
'''
import shutil
import sys, os
from collections import defaultdict
from lib import kscc_unsupervised as us
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)  # Removes all the subdirectories!
        os.makedirs(path)
'''
This function is used to calculate the dissimilarity between 2 attributes x and y at the iattr(d)
using the global_attr_freq in which the global_axttr_freq[i][x] is the frequency of the attribute at
i with value x in the whole samples:
dis(x, y) = 1 - 2 * log(P{x, y}) / (log(P{x}) + log(P{y}))
'''

def attr_dissim(x, y, iattr, global_attr_freq):
    '''
    Dissimilarity between 2 categorical attributes x and y at the attribute iattr, i.e
    dis(x, y) = 1 - 2 * log(P{x, y}) / (log(P{x}) + log(P{y}))
    '''
    if (global_attr_freq[iattr][x] == 1.0) and (global_attr_freq[iattr][y] == 1.0):
        return 0
    if x == y:
        numerator = 2 * math.log(global_attr_freq[iattr][x])
    else:
        numerator = 2 * math.log((global_attr_freq[iattr][x] + global_attr_freq[iattr][y]))
    denominator = math.log(global_attr_freq[iattr][x]) + math.log(global_attr_freq[iattr][y]) #Noted by Tai Dinh, Equation 21, page 124
    return 1 - numerator / denominator

'''
This function is used to calculate the dissimilarity between a centroid and a vector a
'''

def vector_matching_dissim(centroid, a, global_attr_freq):
    '''Get distance between a centroid and a'''
    distance = 0.
    for ic, curc in enumerate(centroid):
        keys = curc.keys()
        for key in keys:
            distance += curc[key] * attr_dissim(key, a[ic], ic, global_attr_freq)
    return distance


'''
This function is used to calculate the distances between centroid clusters and a data point, using the global_attr_freq.
global_axttr_freq[i][x] is the probability of the attribute at position i and value x on the whole samples.
categorical is the set of categorical attributes in the data point.
'''


def vectors_matching_dissim(vectors, a, global_attr_freq):
    '''Get nearest vector in vectors to a'''
    min = np.Inf
    min_clust = -1
    for clust in range(len(vectors)):
        distance = vector_matching_dissim(vectors[clust], a, global_attr_freq)
        if distance < min:
            min = distance
            min_clust = clust
    return min_clust, min


'''
This function is used to transfer a vector point from this cluster (from_clust) to another cluster (to_clust)
ipoint is the index of vector in the samples.
membership[cluster_index, ipoint] = 1 means vector with the index ipoint belongs to the cluster_index.

cl_attr_freq[cluster_index][iattr][curattr] is the frequency of the attribute having value curattr at iattr in cluster cluster_index.
In fact, k are kept in the cl_attr_freq instead of k/N such that k is the number of appearance of attribute at the iattr with value curattr,
N is the number of data objects in the cluster. The reason is k and N are probably change, in this case recalculating k/N is more complex than k. 

Note that global_attr_freq stores frequency (k/N) because it only needs one time to calculate and values keep permanently. 
'''

def move_point_between_clusters(point, ipoint, to_clust, from_clust,
    cl_attr_freq, membership):
    '''Move point between clusters, categorical attributes'''
    membership[to_clust, ipoint] = 1
    membership[from_clust, ipoint] = 0
    # Update frequencies of attributes in clusters
    for iattr, curattr in enumerate(point):
        cl_attr_freq[to_clust][iattr][curattr] += 1
        cl_attr_freq[from_clust][iattr][curattr] -= 1
    return cl_attr_freq, membership

def matching_dissim(a, b):
    '''Simple matching dissimilarity function'''
    return np.sum(a != b, axis=1)

'''
This function is used to calculate the lambda as the formula in slide page 15
cl_attr_freq[iattr][curattr] is the probability of attribute at the iattr having value curattr in cluster clust
clust_members is the number of data objects in the cluster
'''

def cal_lambda(cl_attr_freq, clust_members):
    '''Re-calculate optimal bandwitch for each cluster'''
    if clust_members <= 1:
        return 0.

    numerator = 0.
    denominator = 0.

    for iattr, curattr in enumerate(cl_attr_freq):
        n_ = 0.
        d_ = 0.
        keys = curattr.keys()
        for key in keys:
            n_ += math.pow(1.0 * curattr[key] / clust_members, 2)
            d_ += math.pow(1.0 * curattr[key] / clust_members, 2)
        numerator += (1 - n_)
        denominator += (d_ - 1.0 / (len(keys)))

    # print denominator
    # assert denominator != 0, "How can denominator equal to 0?"
    if clust_members == 1 or denominator == 0:
        return 0
    return (1.0 * numerator) / ((clust_members - 1) * denominator)

def _cal_global_attr_freq(X, npoints, nattrs):
    # global_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of attributes.
    global_attr_freq = [defaultdict(float) for _ in range(nattrs)]

    for ipoint, curpoint in enumerate(X):
        for iattr, curattr in enumerate(curpoint):
            global_attr_freq[iattr][curattr] += 1.
    for iattr in range(nattrs):
        for key in global_attr_freq[iattr].keys():
            global_attr_freq[iattr][key] /= npoints

    return global_attr_freq


'''
This function is used to calculate the centroid center at each attribute.

* ldb is lambda
* cl_attr_freq_attr is cl_attr_freq[clust][iattr], is the number of attribute at the index iattr in the cluster clust.
* clust_members is the number of data objects in the cluster
* global_attr_count is the number of attribute at the index iattr in the whole dataset X.
'''

def cal_centroid_value(lbd, cl_attr_freq_attr, cluster_members, attr_count):
    '''Calculate centroid value at iattr'''
    assert cluster_members >= 1, "Cluster has no member, why?"

    keys = cl_attr_freq_attr.keys()
    vjd = defaultdict(float)
    for odl in keys:
        vjd[odl] = lbd / attr_count + (1 - lbd) * (1.0 * cl_attr_freq_attr[odl] / cluster_members) #Noted by Tai Dinh equation 12, page 121
    return vjd

'''
This function is the loop for the k-CMM algorithm
For each vector curpoint with the index ipoint in X, the purpose is to find the nearest centroid with this vector.
'''

def kscc_kernel_it_iter(X, centroids, cl_attr_freq, membership, global_attr_freq, lbd):
    '''Single iteration of k-representative clustering algorithm'''
    moves = 0
    for ipoint, curpoint in enumerate(X):
        clust, distance = vectors_matching_dissim(centroids, curpoint, global_attr_freq)
        if membership[clust, ipoint]:
            # Sample is already in its right place
            continue

        # Move point and update old/new cluster frequencies and centroids
        '''
        moves is the number of moving vectors between cluster
        old_clust is the old index of vector curpoint
        '''
        moves += 1
        old_clust = np.argwhere(membership[:, ipoint])[0][0]

        '''
        Move vector with index ipoint from old_clust to clust, meanwhile recalculate the probability of attributes in the corresponding clusters.
        '''
        cl_attr_freq, membership = move_point_between_clusters(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membership)

        # In case of an empty cluster, reinitialize with a random point
        # from the largest cluster.
        '''
        After moving vectors from old_clust to new_clust, if the old_clust is empty, 
        then get an arbitrary vector from the largest cluster to this cluster to avoid empty clusters.  
        '''
        if sum(membership[old_clust, :]) == 0:
            from_clust = membership.sum(axis = 1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindex = np.random.choice(choices)

            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, old_clust, from_clust, cl_attr_freq, membership)

        # Re-calculate lambda of changed centroid
        for curc in (clust, old_clust):
            lbd[curc] = cal_lambda(cl_attr_freq[curc], sum(membership[curc, :]))

        # Update new and old centroids by choosing mode of attribute.
        for iattr in range(len(curpoint)):
            for curc in (clust, old_clust):
                cluster_members = sum(membership[curc, :])
                attr_count = len(cl_attr_freq[curc][iattr].keys())
                centroids[curc][iattr] = cal_centroid_value(lbd[curc], cl_attr_freq[curc][iattr], cluster_members, attr_count)
    return centroids, moves, lbd


'''
 This function is used to calculate the sum of distances between vectors inside X and centroids of clusters after each step.
 Labels is the label of vector in X, labels[x] = c means vector with index is x is belonged to the cluster that has its index is c.
 Cost is the sum of dissimilarity.
'''


def _labels(X, centroids, global_attr_freq):
    '''
    Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-modes algorithm.
    '''
    n_points = X.shape[0]
    cost = 0.
    labels = np.empty(n_points, dtype = 'int64')
    for ipoint, curpoint in enumerate(X):
        clust, diss = vectors_matching_dissim(centroids, curpoint, global_attr_freq)
        assert clust != -1, "Why there is no cluster for me?"
        labels[ipoint] = clust
        cost += diss

    return labels, cost

'''
Ramdomly initialize vectors in X into clusters
'''

def _init_clusters(X, centroids, n_clusters, nattrs, npoints, verbose):
    # __INIT_CLUSTER__
    if verbose:
        print("Init: Initalizing clusters")
    membership = np.zeros((n_clusters, npoints), dtype='int64')
    # cl_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute.
    cl_attr_freq = [[defaultdict(int) for _ in range(nattrs)]
                    for _ in range(n_clusters)]
    for ipoint, curpoint in enumerate(X):
        # Initial aassignment to clusterss
        clust = np.argmin(matching_dissim(centroids, curpoint))
        membership[clust, ipoint] = 1
        # Count attribute values per cluster
        for iattr, curattr in enumerate(curpoint):
            cl_attr_freq[clust][iattr][curattr] += 1

    # Move random selected point from largest cluster to empty cluster if exists
    for ik in range(n_clusters):
        if sum(membership[ik, :]) == 0:
            from_clust = membership.sum(axis=1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindex = np.random.choice(choices)
            # Move random selected point to empty cluster
            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, ik, from_clust, cl_attr_freq, membership)

    return cl_attr_freq, membership

def kscc_kernel_it(X, n_clusters, n_init, verbose):
    X = np.asanyarray(X)
    npoints, nattrs = X.shape
    assert n_clusters < npoints, "More clusters than data points?"

    all_centroids = []
    all_labels = []
    all_costs = []

    global global_attr_freq
    global_attr_freq = _cal_global_attr_freq(X, npoints, nattrs)

    for init_no in range(n_init):
        seeds = np.random.choice(range(npoints), n_clusters)
        centroids = X[seeds]
        cl_attr_freq, membership = _init_clusters(X, centroids, n_clusters, nattrs, npoints, verbose)

        centroids = [[defaultdict(float) for _ in range(nattrs)]
                     for _ in range(n_clusters)]

        # Perform initial centroid update
        lbd = np.zeros(n_clusters, dtype='float')
        for ik in range(n_clusters):
            cluster_members = sum(membership[ik, :])
            for iattr in range(nattrs):
                centroids[ik][iattr] = cal_centroid_value(lbd[ik], cl_attr_freq[ik][iattr], cluster_members,
                                                      len(cl_attr_freq[ik][iattr].keys()))
        # __ITERATION__
        if verbose:
            print("Starting iterations...")
        labels = None
        converged = False
        cost = np.Inf
        while not converged:
            if verbose:
                print("...kn_it_local loop")
            centroids, moves, lbd = kscc_kernel_it_iter(X, centroids, cl_attr_freq, membership, global_attr_freq, lbd)
            if verbose:
                print("...Update labels, costs")
            labels, ncost = _labels(X, centroids, global_attr_freq)
            converged = (moves == 0) or (ncost >= cost)
            cost = ncost
            if verbose:
                print("Run {}, moves: {}, cost: {}"
                      .format(init_no + 1, moves, cost))
        # Store result of current runt
        all_centroids.append(centroids)
        all_labels.append(labels)
        all_costs.append(cost)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}, min cost is {}".format(best + 1, all_costs[best]))

    return all_centroids[best], all_labels[best], all_costs[best]

class KSCC_Kernel_IT(object):
    def __init__(self, n_clusters,n_init, verbose=1):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.verbose = verbose

    def fit(self, X, **kwargs):
        '''Compute k-representative clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        '''

        self.cluster_centroids_, self.labels_, self.cost_ = kscc_kernel_it(X, self.n_clusters, self.n_init, self.verbose)
        return self

    def fit_predict(self, X, **kwargs):
        '''Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        '''
        return self.fit(X, **kwargs).labels_

    def predict(self, X, **kwargs):
        '''Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        '''
        # assert hasattr(self, 'cluster_centroids_'), "Model not yet fitted"
        # return _labels_cost(X, self.cluster_centroids_)[0]

def pair_distance(a,b,n_attrs):
    distance = 0.
    for j in range(n_attrs):
            distance += attr_dissim(a[j], b[j], j, global_attr_freq)
    return distance

def pairwise_distances(x):
    n_samples, n_attrs = x.shape
    distance = np.zeros((n_samples,n_samples))
    for row in range(n_samples):
        col = row + 1
        while col < n_samples:
            distance[row,col] = pair_distance(x[row],x[col], n_attrs)
            distance[col,row] = distance[row,col]
            col+=1
    return distance


def run(argv):
    n_init = 100
    ifile = "sake.csv"
    ofile = "kscc_it_sake.csv"
    output_folder = "output"
    makedirs(output_folder)
    delim = ","
    verbose = 0

    # Get samples & labels
    # if not use_first_column_as_label:
    x = np.genfromtxt(ifile, dtype = str, delimiter = delim)[:]
    result = []
    for n_clusters in range(2, 11):
        # Create a subplot with 1 row and 2 columns
        fig1, ax1 = plt.subplots()
        # fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])

        kr = KSCC_Kernel_IT(n_clusters=n_clusters, n_init=n_init, verbose= verbose)
        kr.fit_predict(x)
        cluster_labels = kr.labels_


        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        _distance = pairwise_distances(x)
        sample_silhouette_values, silhouette_avg = us.silhouette_score(x, cluster_labels,_distance)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        result.append(silhouette_avg)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # cmap = cm.get_cmap("Spectral")
            # color = cmap(float(i) / n_clusters)

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # ax1.set_title("The silhouette plot for the various clusters.")
        # ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("#clusters = %d" % n_clusters),
                     fontsize=13)
        plt.savefig("{0}/kscc_it_sake{1}.pdf".format(output_folder, n_clusters), dpi=300)
    # plt.show()
    import csv
    with open("{0}/{1}".format(output_folder,ofile), 'w') as fp:
        out = csv.writer(fp)
        out.writerows(map(lambda x: [x], result))
    print("Finish!!!")


if __name__ == "__main__":
    run(sys.argv[1:])