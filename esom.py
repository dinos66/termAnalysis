# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import somoclu, time
import numpy as np

# codebookini = pd.read_table(prefix + str(2006) + ".txt", sep="\t", header=0,index_col=0)
# U, s, V = np.linalg.svd(codebookini, full_matrices=False)


years = range(2006, 2013)
n_columns, n_rows = 50, 30
savefig = False
prefix = "data/pruned100/undirectedAdjacency_matrix_"

som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid")

for year in years:
    df = pd.read_table(prefix + str(year) + ".txt", sep="\t", header=0,index_col=0)
    dfmax = df.max()
    dfmax[dfmax == 0] = 1
    df = df / dfmax
    labels = df.index.tolist()
    som.update_data(df.values)
    if year == years[0]:
        epochs = 10
        radius0 = 0
        scale0 = 0.1
    else:
        radius0 = n_rows//5
        scale0 = 0.03
        epochs = 3
    if savefig:
        filename = prefix + str(year) + ".png"
    else:
        filename = None


    som.train(epochs=epochs, radius0=radius0, scale0=scale0)
    som.view_umatrix(colormap="Spectral_r", bestmatches=True, labels=labels,filename=filename)
    mycoords = list(som.bmus)
    
    from scipy import spatial
    distMat = spatial.distance.pdist(som.bmus, 'euclidean')
    distMat = spatial.distance.squareform(distMat)

    vv=input('hi')

