fix matrix extraction according to pruned100 file before running 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:
# Purpose:       This .py file produces an analysis of the terms
#
# Required libs: numpy
# Author:        konkonst
#
# Created:       21/08/2015
# Copyright:     (c) ITI (CERTH) 2013
# Licence:       <apache licence 2.0>
#-------------------------------------------------------------------------------
import time, glob,os,pickle, igraph, itertools
import numpy as np

print('extract termanalysis')
print(time.asctime( time.localtime(time.time()) ))

t = time.time()


files = glob.glob('./data/watches_edges_*.txt')

edgeDict = {'terms':[]}
for filename in files:
    year = filename[-8:-4]
    tmpTerms = []
    edgeDict[year] = {}
    with open(filename, 'r') as f:
        print(filename)
        adjList = []
        next(f)
        for line in f:
            x = line.split('\t')
            tripletuple = x[0].split(',')
            edgeDict['terms'].extend(tripletuple)
            tmpTerms.extend(tripletuple)
            tripletuple.append(int(x[1].strip()))
            adjList.append(tripletuple)
        edgeDict[year]['adjList'] = adjList
    print('There are %s unique nodes for year %s' %(len(set(tmpTerms)),year))

edgeDict['terms'] = list(set(edgeDict['terms']))
edgeDict['terms'].sort()
print('There are %s unique nodes globally' %len(edgeDict['terms']))

yearList = []
for filename in files[4:]:
    year = filename[-8:-4]
    yearList.append(year)
    print(year)
    gDirected=igraph.Graph.Full(0, directed = True)
    gDirected.es['weight'] = 1
    gDirected.add_vertices(edgeDict['terms'])
    myEdges = [(x[0],x[1]) for x in edgeDict[year]['adjList']]
    myWeights = [x[2] for x in edgeDict[year]['adjList']]    
    gDirected.add_edges(myEdges)
    gDirected.es["weight"] = myWeights
    edgeDict[year]['graph'] = gDirected
    print('created graph')
    nodes = gDirected.vs['name']
    with open('./data/watches_nodes_'+year+'.txt', 'r') as f:
        edgeDict[year]['term'] = {'Freq':{},'degree':{},'pageRank':{},'normPageRank':{}}
        next(f)
        for line in f:
            x = line.strip().split('\t')
            edgeDict[year]['term']['Freq'][x[0]] = x[1]
            # if x[0] not in nodes:
            #     gDirected.add_vertex(x[0])
    pageRank = gDirected.pagerank(weights = 'weight')
    print('extracted pagerank')
    maxPR = max(pageRank)
    normPageRank = [x/maxPR for x in pageRank]
    for x in nodes:
        edgeDict[year]['term']['pageRank'][x] = pageRank[nodes.index(x)]
        edgeDict[year]['term']['normPageRank'][x] = normPageRank[nodes.index(x)]
        edgeDict[year]['term']['degree'][x] = str(gDirected.degree(x))

    # '''write individual term analysis'''
    # with open('./data/newWatches_nodes_'+year+'.txt', 'w') as d:
    #     d.write('Term\tFreq\tDegree\tPR\tNPR\n')
    #     rankPgRank = sorted(edgeDict[year]['term']['pageRank'], key=edgeDict[year]['term']['pageRank'].get,reverse=True)
    #     for at in rankPgRank:
    #         try:
    #             freq = edgeDict[year]['term']['Freq'][at]
    #         except:
    #             freq = str(0)
    #             pass
    #         tmpline = '\t'.join([at,freq,edgeDict[year]['term']['degree'][at],'{0:.6f}'.format(edgeDict[year]['term']['pageRank'][at]),'{0:.6f}'.format(edgeDict[year]['term']['normPageRank'][at])])
    #         d.write(tmpline+'\n')

    '''creating adjacency mat'''
    print('creating adjacency matrix')
    adjMat = gDirected.get_adjacency(attribute='weight')
    del(gDirected)
    adjMat = np.array(adjMat.data)
    # print('writing adjacency matrix to file')
    # with open('./data/adjacency_matrix_'+year+'.txt', 'w') as d:
    #     d.write('Term\tAdjacency Matrix\n')
    #     for idx,s in enumerate(nodes):
    #         distLine = [str(x) for x in adjMat[idx].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    '''distance matrix extraction'''
    print('estimate distance matrix')
    from scipy import spatial
    distMat = spatial.distance.pdist(adjMat, 'euclidean')
    distMat = spatial.distance.squareform(distMat)
    Yuptri = np.triu(distMat)
    del(adjMat)

    '''Write the distance matrix to a file'''
    print('writing distance matrix to file')
    with open('./data/similarities/distance_matrix_'+year+'.txt', 'w') as d:
        d.write('Term\tEuclidean Distance Matrix\n')
        for idx,s in enumerate(nodes):
            distLine = ['{0:.2f}'.format(x) for x in Yuptri[idx].tolist()]
            d.write(s+'\t'+'\t'.join(distLine)+'\n')
    del(Yuptri)
    
    '''estimate the (PR_userN * PR_userN+1)/distance matrix'''
    print('estimate gravity')
    gravMat = np.zeros(distMat.shape)
    for n,v in edgeDict[year]['term']['pageRank'].items():
        gravMat[nodes.index(n)] = v
    pgrMat = np.zeros(distMat.shape)
    for n,v in edgeDict[year]['term']['pageRank'].items():
        pgrMat[:,nodes.index(n)] = v
    gravMat = np.multiply(gravMat,pgrMat)
    gravMat = (-gravMat*100000)
    gravMat = np.multiply(gravMat,abs(np.identity(gravMat.shape[0])-1))
    gravMat = np.divide(gravMat,(distMat+1))#e-8)


    gravMatTriu = np.triu(gravMat)
    # edgeDict[year]['distMat'] = distMat
    # edgeDict[year]['gravMat'] = gravMat
    del(gravMat,distMat)
    print('writing gravity matrix to file')
    with open('./data/similarities/gravity_matrix_'+year+'.txt', 'w') as d:
        d.write('Term\tGravity Matrix\n')
        for idx,s in enumerate(nodes):
            distLine = ['{0:.6f}'.format(x) for x in gravMatTriu[idx].tolist()]
            d.write(s+'\t'+'\t'.join(distLine)+'\n')

    elapsed = time.time() - t
    print('Elapsed: %.2f seconds' % elapsed)

# '''write pagerank evolution of nodes'''
# print('write pagerank evolution of nodes')
# with open('./data/termPageRankEvolution.txt', 'w') as d:
#     d.write('Term\t2006\t2007\t2008\t2009\t2010\t2011\t2012\n')
#     for x in nodes:
#         pgPerYearList = []
#         for y in yearList:
#             pgPerYearList.append(str(float(edgeDict[y]['term']['pageRank'][x])))
#         d.write(x+'\t'+'\t'.join(pgPerYearList)+'\n')


##pickle.dump(edgeDict,open('./data/pickles/edgeDict.pck','wb'), protocol = 2)
elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
