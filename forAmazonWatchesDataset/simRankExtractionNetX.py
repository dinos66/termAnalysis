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
import time, glob,os,pickle, numpy, itertools,pprint
import igraph
print('testing file')
print(time.asctime( time.localtime(time.time()) ))

t = time.time()

def pruneNodes(G):
    nodes = G.vs['name']
    closeNodes = []
    distanceDict = {}
    preds = {x:G.predecessors(x) for x in nodes if G.predecessors(x)}
    newNodes = list(preds.keys())
    for i in newNodes:
        distanceDict[i] = G.get_shortest_paths(i,mode='ALL')
        closeNodes.extend([(i,nodes[x[-1]]) for x in distanceDict[i] if len(x)>1 and len(x)<5])
##    print(len(set(closeNodes)))
    return set(closeNodes)

def simrankPruned(G, r=0.8, max_iter=20, eps=1e-4):

    nodes = G.vs['name']
    print(len(nodes))
    nodes_idx = {nodes[i]: i for i in range(0, len(nodes))}
    simR_pr = numpy.zeros(len(nodes))
    simR = numpy.identity(len(nodes))
    timeini = time.time()
    prunedNodes = pruneNodes(G)
    preds = {x:G.predecessors(x) for x in nodes}# if G.predecessors(x)}
    newNodes = list(preds.keys())
    print(len(newNodes))

    print('Elapsed: %.2f seconds' % elapsed)
    for i in range(max_iter):
        if numpy.allclose(simR, simR_pr, atol=eps):
            break
        simR_pr = numpy.copy(simR)
        for u, v in prunedNodes:
            if u is v:
                continue
            u_np, v_np = preds[u], preds[v]
            if u_np and v_np:
                S_uv = sum([simR_pr[u_n][v_n] for u_n, v_n in itertools.product(u_np, v_np)])
                simR[nodes_idx[u]][nodes_idx[v]] = (r * S_uv) / (len(u_np) * len(v_np))
    return simR


def simrankFull(G, r=0.8, max_iter=20, eps=1e-4):

    nodes = G.vs['name']
    nodes_idx = {nodes[i]: i for i in range(0, len(nodes))}

    simR_pr = numpy.zeros(len(nodes))
    simR = numpy.identity(len(nodes))
    preds = {x:G.predecessors(x) for x in nodes if G.predecessors(x)}
    newNodes = list(preds.keys())
    print(len(nodes))
    print(len(newNodes))
    allCombinations = list(itertools.product(newNodes,newNodes))
    allCombinations.sort()
    print(len(allCombinations))
    sameCombs = list(zip(newNodes,newNodes))
    sameCombs.sort()
    for s in sameCombs:
        allCombinations.remove(s)
    for i in range(max_iter):
        if numpy.allclose(simR, simR_pr, atol=eps):
            break
        simR_pr = numpy.copy(simR)
        for u, v in allCombinations:
            if u is v:
                continue
            u_np, v_np = preds[u], preds[v]
            S_uv = sum([simR_pr[u_n][v_n] for u_n, v_n in itertools.product(u_np, v_np)])
            simR[nodes_idx[u]][nodes_idx[v]] = (r * S_uv) / (len(u_np) * len(v_np))
    return simR

def euclideanCoords(G,distMat = None):
    euclSpace = G.layout_mds(dist = distMat)
    allCoordinates = euclSpace.coords
    return euclSpace

def laplacianDistance(G):
    laplacianSpace = G.laplacian(weights = 'weight')
    return laplacianSpace


adjList = [('Uni','Prof1',1), ('Uni', 'Prof2',1), ('Prof1','Stud1',1), ('Stud1','Uni',1), ('Prof2', 'Stud2',1), ('Stud2', 'Prof2',1)]
adjList = [('Uni','Prof1',1), ('Uni', 'Prof2',1), ('Prof1','Stud1',1), ('Stud1','Uni',1), ('Prof2', 'Stud2',1), ('Stud2', 'Prof2',1),('Maria','Dinos',2),('Dinos','Maria',3)]
gDirected=igraph.Graph.TupleList(adjList, directed = True, weights=True)

##maxWeight = max([x[2] for x in adjList])
##adjList = [(a,b,c/maxWeight) for a,b,c in adjList]
##t2 = time.time()
##distanceDict = pruneNodes(gDirected)
##pprint.pprint(distanceDict)
##elapsed = time.time() - t2
##print('Elapsed: %.2f seconds' % elapsed)
##k = simrankPruned(gDirected)
##pprint.pprint(k.round(3))
##print(gDirected.vs['name'])
##
##euclideanCoordinates = euclideanDistance(gDirected).coords
##xpl=[round(x[0],4) for x in euclideanCoordinates]
##ypl=[round(x[1],4) for x in euclideanCoordinates]
##from matplotlib import pyplot
##from matplotlib import interactive
##v=list(zip(xpl,ypl))
####pprint.pprint(v)
##colors=['b', 'c', 'y', 'm', 'r']
##labels = gDirected.vs['name']
##pyplot.scatter(xpl,ypl, c=colors)
##for label, x, y in zip(labels, xpl, ypl):
##    pyplot.annotate(label, xy = (x, y), xytext = (-20, 20),
##        textcoords = 'offset points', ha = 'right', va = 'bottom',
##        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
##        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
##interactive(True)
##pyplot.show()
##nodes = gDirected.vs['name']
##for top in nodes:
##    topCoord = euclideanCoordinates[nodes.index(top)]
##    topCoord = numpy.array(topCoord)
##    eDistDict = {}
##    for n in nodes:
##        eDistDict[n] = numpy.linalg.norm(topCoord-numpy.array(euclideanCoordinates[nodes.index(n)]))
##    print(top)
##    pprint.pprint(eDistDict)
##    print('\n')
##
##ebs = gDirected.edge_betweenness()
##sortEB = ebs.copy()
##sortEB.sort(reverse = True)
##connDict = {','.join([nodes[gDirected.es[idx].tuple[0]],nodes[gDirected.es[idx].tuple[1]]]):ebs[idx] for idx, eb in enumerate(ebs)}
##sortConnDict = sorted(connDict, key=connDict.get,reverse = True)
##for s in sortConnDict:
##    print(s+'\t%s' %connDict[s])
##
##adjMat = gDirected.get_adjacency()
##adjMat = numpy.array(adjMat.data)
##from scipy import spatial
##Y = spatial.distance.pdist(adjMat, 'euclidean')
##Y = spatial.distance.squareform(Y)


##xa = laplacianDistance(gDirected)
##print(xa)

##pprint.pprint(gDirected.similarity_dice())
##pprint.pprint(gDirected.similarity_inverse_log_weighted())
##pprint.pprint(gDirected.similarity_jaccard())

#--------------------------------------------------------------
##files = glob.glob('./data/watches_edges_*.txt')
##for filename in files:
##    year = filename[-8:-4]
##    with open(filename, 'r') as f:
##        users = []
##        t1 = time.time()
##        print(filename)
####with open('./testfile.txt', 'r') as f:
##        adjList = []
##        next(f)
##        for line in f:
##            x = line.split('\t')
##            tripletuple = x[0].split(',')
##            users.extend(tripletuple)
##            tripletuple.append(int(x[1].strip()))
##            adjList.append(tripletuple)
##        print('There are %s edges and %s nodes' %(len(adjList),len(set(users))))
##        maxWeight = sum([x[2] for x in adjList])
##        adjList = [(a,b,c/maxWeight) for a,b,c in adjList]
##        print('Extracted adjList')
##        gDirected=igraph.Graph.TupleList(adjList, directed = True, weights=True)
##        print('Created igraph')
##    ##    k = simrankFull(gDirected)
##    ##    elapsed = time.time() - t1
##    ##    print('simrank Elapsed: %.2f seconds' % elapsed)
##        nodes = gDirected.vs['name']
##
####        adjMat = gDirected.get_adjacency()
####        adjMat = numpy.array(adjMat.data)
####        from scipy import spatial
####        Y = spatial.distance.pdist(adjMat, 'euclidean')
####        distMat = spatial.distance.squareform(Y)
####        print('Extracted euclidean distance matrix')
##
##        xa = euclideanCoords(gDirected)#,distMat)
##        print('Mapped nodes to euclidean space')
##        xpl=[x[0] for x in xa.coords]
####        xpl=[round(x[0],6) for x in xa.coords]
##        minXpl = min(xpl)
##        if minXpl < 0:
##            aminXpl = abs(minXpl)
##            xpl = [x+aminXpl+1 for x in xpl]
##        ypl=[x[1] for x in xa.coords]
####        ypl=[round(x[1],6) for x in xa.coords]
##        minYpl = min(ypl)
##        if minYpl < 0:
##            aminYpl = abs(minYpl)
##            ypl = [y+aminYpl+1 for y in ypl]
##        from matplotlib import pyplot
##        from matplotlib import interactive
##        fig = pyplot.figure()
##        ax = pyplot.gca()
##        ax.scatter(xpl,ypl)
##        ax.set_ylim(min(ypl)-1,max(ypl)+1)
##        ax.set_xlim(min(xpl)-1,max(xpl)+1)
##        labels = gDirected.vs['name']
##        for label, x, y in zip(labels, xpl, ypl):
##            pyplot.annotate(label, xy = (x, y), xytext = (-10, 10),
##                textcoords = 'offset points', ha = 'right', va = 'bottom',
##                bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.5),
##                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
##        interactive(True)
##        pyplot.show()
##        try:
##            year
##        except:
##            year = 'test'
##            pass
##        pyplot.savefig('./images/'+str(year)+'_euclSpaceMapping_via_shortestPaths.jpg', bbox_inches='tight', format='jpg')
####        pyplot.savefig('./images/'+str(year)+'_euclSpaceMapping_via_distMatrixSumnormed.jpg', bbox_inches='tight', format='jpg')
##        pyplot.close()
##
##        elapsed = time.time() - t1
##        print('test1 Elapsed: %.2f seconds' % elapsed)
####
####    ebs = gDirected.edge_betweenness(directed=True,weights='weight')
####    sortEB = ebs.copy()
####    sortEB.sort(reverse = True)
####    connDict = {','.join([nodes[gDirected.es[idx].tuple[0]],nodes[gDirected.es[idx].tuple[1]]]):ebs[idx] for idx, eb in enumerate(ebs)}
####    sortConnDict = sorted(connDict, key=connDict.get,reverse = True)
####    for s in sortConnDict[:50]:
####        print(s+'\t%s' %connDict[s])
########
########    simD = gDirected.similarity_dice()
########    simI = gDirected.similarity_inverse_log_weighted()
########    simJ = gDirected.similarity_jaccard()

elapsed = time.time() - t
print('test1 Elapsed: %.2f seconds' % elapsed)
