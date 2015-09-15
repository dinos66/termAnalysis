#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:
# Purpose:       This .py file creates a graph and extracts simrank
#
# Required libs: numpy
# Author:        konkonst
#
# Created:       20/08/2013
# Copyright:     (c) ITI (CERTH) 2013
# Licence:       <apache licence 2.0>
#-------------------------------------------------------------------------------
import time, glob,os,pickle, igraph, numpy, itertools

print('extract simrank')
print(time.asctime( time.localtime(time.time()) ))

t = time.time()


def simrankPruned(G,topList, r=0.8, max_iter=20, eps=1e-4):
    t = time.time()
    nodes = G.vs['name']
    print(len(nodes))
    nodes_idx = {nodes[i]: i for i in range(0, len(nodes))}
    simR_pr = numpy.zeros(len(nodes))
    simR = numpy.identity(len(nodes))
    allCombinations = list(itertools.product(topList,nodes))
    allCombinations.extend(list(itertools.product(nodes,topList)))
    allCombinations = list(set(allCombinations))
    print('allCombs: %s' %len(allCombinations))

    preds = {x:G.predecessors(x) for x in nodes}
    for i in range(max_iter):
        print(i)
        if numpy.allclose(simR, simR_pr, atol=eps):
            break
        simR_pr = numpy.copy(simR)
        for u, v in allCombinations:
            if u is v:
                continue
            u_np, v_np = preds[u], preds[v]
            if u_np and v_np:
                tmpCombs = itertools.product(u_np, v_np)
                lenProd = len(u_np) * len(v_np)
                S_uv = sum([simR_pr[u_n][v_n] for u_n, v_n in tmpCombs])
                simR[nodes_idx[u]][nodes_idx[v]] = (r * S_uv) / lenProd
    return simR


files = glob.glob('./data/watches_edges_*.txt')

edgeDict = {}
for filename in files:
    year = filename[-8:-4]
    edgeDict[year] = {}
    with open(filename, 'r') as f:
        print(filename)
        adjList = []
        users = []
        next(f)
        for line in f:
            x = line.split('\t')
            tripletuple = x[0].split(',')
            users.extend(tripletuple)
            tripletuple.append(int(x[1].strip()))
            adjList.append(tripletuple)
    print('There are %s edges and %s nodes' %(len(adjList),len(set(users))))
    gDirected=igraph.Graph.TupleList(adjList, directed = True, weights=True)
    edgeDict[year]['adjList'] = adjList
    edgeDict[year]['graph'] = gDirected
    nodes = gDirected.vs['name']
    with open('./data/'+year+'_node_centric_importances.txt', 'r') as f:
        topList = []
        for line in f:
            x = line.strip().split('\t')
            if x[0]:
                if x[0] == '#':
                    topNode = [x[1]]
                    topList.append(topNode[0])
    del(topNode)
    edgeDict[year]['simRank'] = simrankPruned(gDirected,topList)
    #write it all down
    with open('./data/'+year+'_simRankSimilarityTopList.txt', 'w') as f:
        for top in topList:
            f.write('\n'+'#\t'+top+'\n')
            secs = combinationDict[top]
            for s in secs:
                f.write(s+'\t'+str(edgeDict[year]['simRank'][nodes.index(top)][nodes.index(s)])+'\n')

pickle.dump(edgeDict,open('./data/pickles/watches_SimRankDict.pck','wb'), protocol = 2)

elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
