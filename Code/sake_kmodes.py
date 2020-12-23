#!/usr/bin/env python
'''Noted by Tai Dinh
This file is used to run k-modes with Silhouette Coefficient
'''
import sys
import numpy as np
from lib import kmodes as km
from lib import kscc_unsupervised as us
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import sys, os


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)  # Removes all the subdirectories!
        os.makedirs(path)

def matching_dissim_silhouette(a, b):
    '''Simple matching dissimilarity function'''
    return np.sum(a != b)

def pairwise_distances(x):
    n_samples, n_attrs = x.shape
    distance = np.zeros((n_samples,n_samples))
    for row in range(n_samples):
        col = row + 1
        while col < n_samples:
            distance[row,col] = matching_dissim_silhouette(x[row],x[col])
            distance[col,row] = distance[row,col]
            col+=1
    return distance

def run(argv):
    n_init = 100
    ifile = "sake.csv"
    ofile = "kmodes_sake.csv"
    output_folder = "output"
    makedirs(output_folder)
    verbose = 0
    delim = ","

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

        kr = km.KModes(n_clusters=n_clusters, init='Huang', n_init=n_init, verbose=verbose)
        kr.fit_predict(x)
        cluster_labels = kr.labels_

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        _distance = pairwise_distances(x)
        sample_silhouette_values, silhouette_avg = us.silhouette_score(x, cluster_labels, _distance)
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
        plt.savefig("{0}/kmodes_sake{1}.pdf".format(output_folder, n_clusters), dpi=300)
    # plt.show()
    import csv
    with open("{0}/{1}".format(output_folder,ofile), 'w') as fp:
        out = csv.writer(fp)
        out.writerows(map(lambda x: [x], result))
        # writer = csv.writer(fp, delimiter=',')
        # writer.writerows(result)


if __name__ == "__main__":
    run(sys.argv[1:])