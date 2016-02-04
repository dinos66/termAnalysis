#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:
# Purpose:       This .py file creates a graph and extracts similarities between nodes
#
# Required libs: numpy
# Author:        konkonst
#
# Created:       20/08/2013
# Copyright:     (c) ITI (CERTH) 2013
# Licence:       <apache licence 2.0>
#-------------------------------------------------------------------------------
import time, glob,os,pickle, igraph, numpy, itertools

print('extract similarities')
print(time.asctime( time.localtime(time.time()) ))

t = time.time()

files = glob.glob('./data/watches_edges_*.txt')

def euclideanDistance(G):
    euclSpace = G.layout_mds()
    allCoordinates = euclSpace.coords
    return allCoordinates

for filename in files:
    year = filename[-8:-4]
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
    nodes = gDirected.vs['name']

    with open('./data/'+year+'_node_centric_importances.txt', 'r') as f:
        allCombinations = []
        combinationDict = {}
        topList = []
        for line in f:
            x = line.strip().split('\t')
            if x[0]:
                if x[0] == '#':
                    topNode = [x[1]]
                    topList.append(topNode[0])

    # t1 = time.time()
    # similarity_dice = gDirected.similarity_dice()
    # with open('./data/similarities/'+year+'_diceSimilarity.txt', 'w') as f:
    #     for top in topList:
    #         f.write('\n'+'#\t'+top+'\n')
    #         valDict = {k:similarity_dice[nodes.index(top)][nodes.index(k)] for k in nodes}
    #         rankedVal = sorted(valDict, key = valDict.get, reverse = True)
    #         for s in rankedVal[:200]:
    #             f.write(s+'\t'+str(valDict[s])+'\n')
    # elapsed = time.time() - t1
    # print('similarity_dice: %.2f seconds' % elapsed)
    # del(similarity_dice)

    # t1 = time.time()
    # similarity_inverse_log_weighted = gDirected.similarity_inverse_log_weighted()
    # with open('./data/similarities/'+year+'_inverse_log_weightedSimilarity.txt', 'w') as f:
    #     for top in topList:
    #         f.write('\n'+'#\t'+top+'\n')
    #         valDict = {k:similarity_inverse_log_weighted[nodes.index(top)][nodes.index(k)] for k in nodes}
    #         rankedVal = sorted(valDict, key = valDict.get, reverse = True)
    #         for s in rankedVal[:200]:
    #             f.write(s+'\t'+str(valDict[s])+'\n')
    # elapsed = time.time() - t1
    # print('similarity_inverse_log_weighted: %.2f seconds' % elapsed)
    # del(similarity_inverse_log_weighted)

    # t1 = time.time()
    # similarity_jaccard = gDirected.similarity_jaccard()
    # with open('./data/similarities/'+year+'_jaccardSimilarity.txt', 'w') as f:
    #     for top in topList:
    #         f.write('\n'+'#\t'+top+'\n')
    #         valDict = {k:similarity_jaccard[nodes.index(top)][nodes.index(k)] for k in nodes}
    #         rankedVal = sorted(valDict, key = valDict.get, reverse = True)
    #         for s in rankedVal[:200]:
    #             f.write(s+'\t'+str(valDict[s])+'\n')
    # elapsed = time.time() - t1
    # print('similarity_jaccard: %.2f seconds' % elapsed)
    # del(similarity_jaccard)

    t1 = time.time()
    euclideanCoordinates = euclideanDistance(gDirected)
    with open('./data/similarities/'+year+'_euclideanSpaceDistance.txt', 'w') as f:
        for top in topList:
            f.write('\n'+'#\t'+top+'\n')
            topCoord = euclideanCoordinates[nodes.index(top)]
            topCoord = numpy.array(topCoord)
            euclDict = {}
            for n in nodes:
                euclDict[n] = numpy.linalg.norm(topCoord-numpy.array(euclideanCoordinates[nodes.index(n)]))
            rankEuclDict = sorted(euclDict, key = euclDict.get)
            for s in rankEuclDict[:200]:
                f.write(s+'\t'+str(euclDict[s])+'\n')
    elapsed = time.time() - t1
    print('similarity_jaccard: %.2f seconds' % elapsed)


elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
