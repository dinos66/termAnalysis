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
import time, glob,os,pickle, igraph, numpy, itertools

print('extract edge centrality')
print(time.asctime( time.localtime(time.time()) ))

t = time.time()


files = glob.glob('./data/watches_edges_*.txt')

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
    
    '''edge_betweenness extraction'''
    ebs = gDirected.edge_betweenness(directed=True,weights='weight')
    sortEB = ebs.copy()
    sortEB.sort(reverse = True)
    connDict = {','.join([nodes[gDirected.es[idx].tuple[0]],nodes[gDirected.es[idx].tuple[1]]]):ebs[idx] for idx, eb in enumerate(ebs)}
    sortConnDict = sorted(connDict, key=connDict.get,reverse = True)
    with open('./data/watches_links_'+year+'.txt', 'w') as d:
        d.write('Term\tBC\n')
        for s in sortConnDict:
            tmpline = '\t'.join([s,'{0:.6f}'.format(connDict[s])])
            d.write(tmpline + '\n')

    elapsed = time.time() - t
    print('Elapsed: %.2f seconds' % elapsed)

elapsed = time.time() - t
print('Total Time Elapsed: %.2f seconds' % elapsed)