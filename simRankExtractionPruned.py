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
'''Parameters'''
radius = 3#shortest path cutoff

def pruneNodes(G,radius,topList):
    nodes = G.vs['name']
    closeNodes = []
    preds = {x:G.predecessors(x) for x in nodes if G.predecessors(x)}
    newNodes = list(preds.keys())
    for idx,i in enumerate(topList):
        closeNodes.extend([(i,nodes[x[-1]]) for x in G.get_shortest_paths(i,mode='ALL') if len(x)>1 and len(x)<radius and nodes[x[-1]] in newNodes])
        if not idx%100:
            print(idx)
            closeNodes = list(set(closeNodes))
            print(len(closeNodes))
    revClNod = [(y,x) for x,y in closeNodes]
    closeNodes.extend(revClNod)
    closeNodes = list(set(closeNodes))
    print('last closeNodes: %s' %len(set(closeNodes)))
    return set(closeNodes)

def simrankPruned(G, toplist, radius, r=0.8, max_iter=20, eps=1e-4,):
    t = time.time()
    nodes = G.vs['name']
    print(len(nodes))
    nodes_idx = {nodes[i]: i for i in range(0, len(nodes))}
    simR_pr = numpy.zeros(len(nodes))
    simR = numpy.identity(len(nodes))
    timeini = time.time()
    prunedNodes = pruneNodes(G,radius,topList)

    allCombinations = list(itertools.product(nodes,nodes))
    print('allcomb: %s vs pruned: %s' %(len(allCombinations),len(prunedNodes)))
    elapsed = time.time() - t
    print('prunedNodes: %.2f seconds' % elapsed)
    preds = {x:G.predecessors(x) for x in nodes}# if G.predecessors(x)}
    # newNodes = list(preds.keys())
    for i in range(max_iter):
        print('Iteration: %s' %i)
        if numpy.allclose(simR, simR_pr, atol=eps):
            break
        simR_pr = numpy.copy(simR)
        for u, v in prunedNodes:
            if u is v:
                continue
            u_np, v_np = preds[u], preds[v]
##            if u_np and v_np:
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
    edgeDict[year]['simRank'] = simrankPruned(gDirected,topList,radius)
    #write it all down
    with open('./data/similarities/'+year+'_simRankSimilarity_Top_Pruned.txt', 'w') as f:
        for top in topList:
            f.write('\n'+'#\t'+top+'\n')
            valDict = {k:edgeDict[year]['simRank'][nodes.index(top)][nodes.index(k)] for k in nodes}
            rankedVal = sorted(valDict, key = valDict.get, reverse = True)
            for s in rankedVal[:200]:
                f.write(s+'\t'+str(valDict[s])+'\n')

    pickle.dump(edgeDict,open('./data/pickles/watches_SimRankDict_Top_Pruned_'+str(year)+'.pck','wb'), protocol = 2)

elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
