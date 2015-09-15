#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:
# Purpose:       This .py file produces an analysis of the terms by pruning the terms according to their
#                consistency over time (show up in all years)
#
# Required libs: numpy
# Author:        konkonst
#
# Created:       21/08/2015
# Copyright:     (c) ITI (CERTH) 2013
# Licence:       <apache licence 2.0>
#-------------------------------------------------------------------------------
import time, glob,os,pickle, igraph, numpy, itertools, collections

print('extract termanalysis pruned')
print(time.asctime( time.localtime(time.time()) ))

t = time.time()

files = glob.glob('./data/watches_edges_*.txt')
try:
    edgeDict = pickle.load(open('./data/pruned/pickles/edgeDictPruned.pck','rb'))
except:
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
    pass


print('There are %s unique nodes globally' %len(edgeDict['terms']))

yearList = []
for filename in files:
    year = filename[-8:-4]
    yearList.append(year)
    print(year)
    try:
        gDirected = edgeDict[year]['graph']
    except:        
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
        pass
    print('created graph')
    nodes = gDirected.vs['name']
    with open('./data/watches_nodes_'+year+'.txt', 'r') as f:
        edgeDict[year]['term'] = {'Freq':{},'degree':{},'pageRank':{},'normPageRank':{}}
        next(f)
        for line in f:
            x = line.strip().split('\t')
            edgeDict[year]['term']['Freq'][x[0]] = x[1]
    pageRank = gDirected.pagerank(weights = 'weight')
    print('extracted pagerank')
    maxPR = max(pageRank)
    normPageRank = [x/maxPR for x in pageRank]
    for x in nodes:
        edgeDict[year]['term']['pageRank'][x] = pageRank[nodes.index(x)]
        edgeDict[year]['term']['normPageRank'][x] = normPageRank[nodes.index(x)]
        edgeDict[year]['term']['degree'][x] = str(gDirected.degree(x))

    # with open('./data/pruned/watches_nodes_'+year+'.txt', 'w') as d:
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

    '''write adjacency mat to file'''
    print('creating adjacency matrix')
    adjMat = gDirected.get_adjacency(attribute='weight')
    del(gDirected)
    adjMat = numpy.array(adjMat.data)
    # print('writing adjacency matrix to file')
    # with open('./data/pruned/adjacency_matrix_'+year+'.txt', 'w') as d:
    #     d.write('Term\tAdjacency Matrix\n')
    #     for idx,s in enumerate(nodes):
    #         distLine = [str(x) for x in adjMat[idx].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    '''distance matrix extraction'''
    print('estimate distance matrix')
    from scipy import spatial
    Y = spatial.distance.pdist(adjMat, 'euclidean')
    Y = spatial.distance.squareform(Y)
    Yuptri = numpy.triu(Y)
    del(adjMat)

    # '''Write the distance matrix to a file'''
    # print('writing distance matrix to file')
    # with open('./data/pruned/similarities/distance_matrix_'+year+'.txt', 'w') as d:
    #     d.write('Term\tEuclidean Distance Matrix\n')
    #     for idx,s in enumerate(nodes):
    #         distLine = [str(round(x,2)) for x in Yuptri[idx].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')
    del(Yuptri)
    
    '''Write the (PR_userN * PR_userN+1)/distance matrix to a file'''
    print('estimate gravity')
    gravMat = numpy.zeros(Y.shape)
    for n,v in edgeDict[year]['term']['normPageRank'].items():
        gravMat[nodes.index(n)] = v
    pgrMat = numpy.zeros(Y.shape)
    for n,v in edgeDict[year]['term']['normPageRank'].items():
        pgrMat[:,nodes.index(n)] = v
    gravMat *= pgrMat
    gravMat = (-gravMat*100)
    gravMat *= abs(numpy.identity(gravMat.shape[0])-1)
    gravMat/=(Y+1)#e-8)


    gravMatTriu = numpy.triu(gravMat)
    # edgeDict[year]['distMat'] = Y
    # edgeDict[year]['gravMat'] = gravMat
    del(gravMat,Y)
    print('writing gravity matrix to file')
    with open('./data/pruned2/similarities/gravity_matrix_'+year+'.txt', 'w') as d:
        d.write('Term\tGravity Matrix\n')
        for idx,s in enumerate(nodes):
            # if 0.00000000103
            distLine = [str(round(x,6)).replace('.',',') for x in gravMatTriu[idx].tolist()]
            d.write(s+'\t'+'\t'.join(distLine)+'\n')

##    print('write grav mat to excel')
##    import xlsxwriter
##    # Create an new Excel file and add a worksheet.
##    workbook = xlsxwriter.Workbook('./data/pruned2/similarities/gravity_matrix_'+year+'xlsx')
##    worksheet = workbook.add_worksheet()
##    for 
##        worksheet.write(2, 0, 123)
##
##    workbook.close()

    elapsed = time.time() - t
    print('Elapsed: %.2f seconds' % elapsed)

# '''write pagerank evolution of nodes'''
# print('write pagerank evolution of nodes')
# with open('./data/pruned/nodePageRankEvolution.txt', 'w') as d:
#     d.write('Term\t2006\t2007\t2008\t2009\t2010\t2011\t2012\n')
#     for x in nodes:
#         pgPerYearList = []
#         for y in yearList:
#             pgPerYearList.append('{0:.6f}'.format(edgeDict[y]['term']['pageRank'][x]))
#         d.write(x+'\t'+'\t'.join(pgPerYearList)+'\n')


pickle.dump(edgeDict,open('./data/pruned/pickles/edgeDictPruned.pck','wb'), protocol = 2)
elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
