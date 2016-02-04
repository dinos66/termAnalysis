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
import time, glob,os,pickle, igraph, numpy, itertools, collections

print('extract simrank persistent')
print(time.asctime( time.localtime(time.time()) ))

t = time.time()
'''Parameters'''
radius = 3#shortest path cutoff

def pruneNodes(G,radius):
    nodes = G.vs['name']
    closeNodes = []
    preds = {x:G.predecessors(x) for x in nodes if G.predecessors(x)}
    newNodes = list(preds.keys())
    for idx,i in enumerate(newNodes):
        closeNodes.extend([(i,nodes[x[-1]]) for x in G.get_shortest_paths(i,mode='IN') if len(x)>1 and len(x)<radius and nodes[x[-1]] in newNodes])
        if not idx%100:
            print(idx)
            closeNodes = list(set(closeNodes))
            print(len(closeNodes))
    revClNod = [(y,x) for x,y in closeNodes]
    closeNodes.extend(revClNod)
    closeNodes = list(set(closeNodes))
    print('last closeNodes: %s' %len(set(closeNodes)))
    return set(closeNodes)

def simrankPruned(G, radius, r=0.8, max_iter=20, eps=1e-4,):
    t = time.time()
    nodes = G.vs['name']
    print(len(nodes))
    nodes_idx = {nodes[i]: i for i in range(0, len(nodes))}
    simR_pr = numpy.zeros(len(nodes))
    simR = numpy.identity(len(nodes))
    timeini = time.time()
    prunedNodes = pruneNodes(G,radius)

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

edgeDict = {'terms':[]}
termsYears = []
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
    termsYears.append(list(set(tmpTerms)))
    print('There are %s unique nodes for year %s' %(len(set(tmpTerms)),year))

repetitiveTerms = collections.Counter(list(itertools.chain.from_iterable(termsYears)))
edgeDict['terms'] = [x for x,v in repetitiveTerms.items() if v == 7]
edgeDict['terms'].sort()
print('There are %s unique nodes globally' %len(edgeDict['terms']))

yearList = []
for filename in files:
    year = filename[-8:-4]
    yearList.append(year)
    print(year)
    gDirected=igraph.Graph.Full(0, directed = True)
    gDirected.es['weight'] = 1
    gDirected.add_vertices(edgeDict['terms'])
    print('Full No of edges: %s' %len(edgeDict[year]['adjList']))
    myEdges = [(x[0],x[1]) for x in edgeDict[year]['adjList'] if x[0] in edgeDict['terms'] and x[1] in edgeDict['terms']]
    myWeights = [x[2] for x in edgeDict[year]['adjList'] if x[0] in edgeDict['terms'] and x[1] in edgeDict['terms']]    
    print('Pruned No of edges %s' %len(myEdges))
    gDirected.add_edges(myEdges)
    gDirected.es["weight"] = myWeights
    edgeDict[year]['graph'] = gDirected
    print('created graph')
    nodes = gDirected.vs['name']
    edgeDict[year]['simRank'] = simrankPruned(gDirected,radius)
    #write it all down
    with open('./data/similarities/'+year+'_simRankSimilarity_Persistent.txt', 'w') as f:
        f.write('term\tsimrankMatrix\n')
        for top in nodes:
            distLine = ['{0:.6f}'.format(x) for x in edgeDict[year]['simRank'][nodes.index(top)].tolist()]
            f.write(top+'\t'+'\t'.join(distLine)+'\n')

    pickle.dump(edgeDict,open('./data/pickles/watches_SimRankDict_Persistent_'+str(year)+'.pck','wb'), protocol = 2)

elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
