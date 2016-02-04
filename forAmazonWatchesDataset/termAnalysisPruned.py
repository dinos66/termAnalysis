don't use unless corrected with pruned100
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
import time, glob,os,pickle, igraph, itertools, collections
import matplotlib.pylab as plt
import pandas as pd    
from matplotlib import interactive
import numpy as np
from scipy.spatial import distance

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
        myEdges,myWeights = [], []
        for x in edgeDict[year]['adjList']:
            if x[0] in edgeDict['terms'] and x[1] in edgeDict['terms']:
                myEdges.append((x[0],x[1]))
                myWeights.append(x[2])   
        print('Pruned No of edges %s' %len(myEdges))

        '''Write pairs of users to txt file for Gephi'''  
        print('writing gephi file')          
        with open('./data/forGephi/prunedWatchesEdges_' + year +'.txt', 'w') as geph:
            geph.write('Source,Target,Weight' + '\n')
            for idg,edg in enumerate(myEdges):
                towrite = list(edg)
                towrite.append(str(myWeights[idg]))
                geph.write(','.join(towrite) + '\n')
        #-------------------
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
        edgeDict[year]['term']['degree'][x] = gDirected.degree(x)

    # pertick = 50
    # topTermsNum = 1000
    # degreeRankedTermsList = sorted(edgeDict[year]['term']['degree'], key = edgeDict[year]['term']['degree'].get,reverse=True)#[:topTermsNum]#[50:2050]
    # degreeData = [edgeDict[year]['term']['degree'][x] for x in degreeRankedTermsList]

    # '''writing term stats to file'''
    # with open('./data/pruned/watches_nodes_'+year+'.txt', 'w') as d:
    #     d.write('Term\tDegree\tPR\tNPR\n')
    #     for at in nodes:
    #         tmpline = '\t'.join([at,str(edgeDict[year]['term']['degree'][at]),'{0:.8f}'.format(edgeDict[year]['term']['pageRank'][at]),'{0:.8f}'.format(edgeDict[year]['term']['normPageRank'][at])])
    #         d.write(tmpline+'\n')

    # '''creating directed adjacency mat'''
    # print('creating adjacency matrix')
    # adjMat = gDirected.get_adjacency(attribute='weight')
    # adjMat = np.array(adjMat.data)
    # print('writing directed adjacency matrix to file')
    # with open('./data/pruned/directedAdjacency_matrix_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(nodes)+'\n')
    #     for s in nodes:
    #         distLine = [str(x) for x in adjMat[nodes.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')
    # del(adjMat)

    # edgeDict[year]['directedAdjMat'] = adjMat

    '''creating undirected adjacency mat'''
    print('creating undirected adjacency matrix')
    unDirected = gDirected.as_undirected(combine_edges='sum')
    # del(gDirected)
    undirectedAdjMat = np.array(unDirected.get_adjacency(attribute='weight').data)
    del(unDirected)      
    # print('writing undirected adjacency matrix to file')
    # with open('./data/pruned/undirectedAdjacency_matrix_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(nodes)+'\n')
    #     for s in nodes:
    #         distLine = [str(x) for x in undirectedAdjMat[nodes.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    edgeDict[year]['undirectedAdjMat'] = undirectedAdjMat

    '''symmetrical distance matrix extraction'''
    print('estimate symmetrical distance matrix')
    from scipy import spatial
    distMat = spatial.distance.pdist(undirectedAdjMat, 'euclidean')
    distMat = spatial.distance.squareform(distMat)
    del(undirectedAdjMat)

    # euclSpaceMapp(gDirected,distMat,nodes)#create euclidean space mapping images

    # '''Write the distance matrix to a file'''
    # print('writing distance matrix to file')
    # with open('./data/pruned/distance_matrix_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(nodes)+'\n')
    #     for s in nodes:
    #         distLine = [str(float(x)) for x in distMat[nodes.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    # np.savez_compressed('./data/pruned/distance_matrix_'+year+'.npz',distMat=distMat)
    
    '''estimate the (PR_userN * PR_userN+1)/distance matrix'''
    normMethod = 'SUM'#normalization method of distance matrix
    if normMethod == 'MAX':
        PRversion = 'normPageRank'
    elif normMethod == 'SUM':
        PRversion = 'pageRank'
    print('estimate gravity')
    gravMat = np.zeros(distMat.shape)
    for n,v in edgeDict[year]['term'][PRversion].items():
        gravMat[nodes.index(n)] = v
    pgrMat = np.zeros(distMat.shape)
    for n,v in edgeDict[year]['term'][PRversion].items():
        pgrMat[:,nodes.index(n)] = v
    gravMat = np.multiply(gravMat,pgrMat)
    PRprodArray = gravMat.reshape(-1)
    gravMat = (-gravMat)#*1000)
    distMat = (distMat+1)/distMat.sum()#make sure this complies with the normMethod
    # gravMat = np.divide(gravMat,distMat)#e-8)
    # gravMat = np.multiply(gravMat,abs(np.identity(gravMat.shape[0])-1))    
    
    '''estimate and write the gravitational potential associated with mass distribution'''
    print('estimate and write the gravitational potential associated with mass distribution')
    with open('./data/pruned/mass_distr_grav_potential_'+normMethod+'normed_'+year+'.txt', 'w') as d:
        d.write('term\tpotential associated with a mass distribution\n') 
        for s in nodes:
            pmd = 0
            for i in nodes:
                pmd += -edgeDict[year]['term'][PRversion][i]/distMat[nodes.index(s)][nodes.index(i)]
            d.write(s+'\t'+str(float(pmd))+'\n')

    # print('Max grav is %s and min grav is %s' %(gravMat.max(),gravMat.min()))
    # print('Max distance is %s and min distance is %s' %(distMat.max(),distMat.min()))
    # del(distMat)
    # print('writing gravity matrix to file')
    # with open('./data/pruned/gravity_matrix_'+normMethod+'normed_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(nodes)+'\n')
    #     for s in nodes:
    #         distLine = [str(float(x)).replace('.',',') for x in gravMat[nodes.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    # np.savez_compressed('./data/pruned/gravity_matrix_'+normMethod+'normed_'+year+'.npz',gravMat=gravMat)

    elapsed = time.time() - t
    print('Elapsed: %.2f seconds' % elapsed)

# '''write pagerank evolution of nodes'''
# print('write pagerank evolution of nodes')
# with open('./data/pruned/termPageRankEvolution.txt', 'w') as d:
#     d.write('Term\t2006\t2007\t2008\t2009\t2010\t2011\t2012\n')
#     for x in nodes:
#         pgPerYearList = []
#         for y in yearList:
#             pgPerYearList.append(str(float(edgeDict[y]['term']['pageRank'][x])))
#         d.write(x+'\t'+'\t'.join(pgPerYearList)+'\n')

# '''compute displacement, velocity and acceleration of terms'''
# termKinetics = {x:{'displacement':[],'velocity':[],'accel':[]} for x in nodes}
# for s in nodes:
#     for y in yearList[1:]:
#         termKinetics[s]['displacement'].append(distance.euclidean(edgeDict[y]['undirectedAdjMat'][nodes.index(s)],edgeDict[str(int(y)-1)]['undirectedAdjMat'][nodes.index(s)]))
#     termKinetics[s]['velocity'] = np.diff(termKinetics[s]['displacement'])
#     termKinetics[s]['accel'] = np.diff(termKinetics[s]['velocity'])
# '''write kinetics of terms'''
# print('write kinetics of terms')
# with open('./data/pruned/term_displacement.txt', 'w') as d:
#     d.write('Term\t2007\t2008\t2009\t2010\t2011\t2012\n')
#     with open('./data/pruned/term_velocity.txt', 'w') as v:
#         with open('./data/pruned/term_acceleration.txt', 'w') as a:
#             for s in nodes:
#                 displLine = [str(float(x)) for x in termKinetics[s]['displacement']]
#                 velLine = [str(float(x)) for x in termKinetics[s]['velocity'].tolist()]
#                 accLine = [str(float(x)) for x in termKinetics[s]['accel'].tolist()]
#                 d.write(s+'\t'+'\t'.join(displLine)+'\n')
#                 v.write(s+'\t'+'\t'.join(velLine)+'\n')
#                 a.write(s+'\t'+'\t'.join(accLine)+'\n')

# pickle.dump(edgeDict,open('./data/pruned/pickles/edgeDictPruned.pck','wb'), protocol = 2)
elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
