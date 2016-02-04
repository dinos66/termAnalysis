#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:
# Purpose:       This .py file produces an analysis of the terms by pruning the terms according to their
#                consistency over time (show up in all years)
#
# Required libs: np
# Author:        konkonst
#
# Created:       21/08/2015
# Copyright:     (c) ITI (CERTH) 2013
# Licence:       <apache licence 2.0>
#-------------------------------------------------------------------------------
import time, glob,os,pickle, igraph, itertools, collections, pprint, somoclu
import matplotlib.pylab as plt
import pandas as pd
from matplotlib import interactive
from matplotlib import pyplot
import numpy as np
from scipy.spatial import distance

print('extract termanalysis pruned for top100 terms')
print(time.asctime( time.localtime(time.time()) ))

t = time.time()



def euclideanCoords(G,distMat = None):
    euclSpace = G.layout_mds(dist = distMat)
    allCoordinates = euclSpace.coords
    return allCoordinates

def euclSpaceMapp(gDirected,distMat,top100List,top100ListIdxs):
    print('extract euclidean space mapping')
    allCoordinates = euclideanCoords(gDirected,distMat)
    print('Mapped nodes to euclidean space')
    xpl=[x[0] for x in allCoordinates]
    minXpl = min(xpl)
    if minXpl < 0:
       aminXpl = abs(minXpl)
       xpl = np.array([x+aminXpl+1 for x in xpl])
    ypl=[x[1] for x in allCoordinates]
    minYpl = min(ypl)
    if minYpl < 0:
       aminYpl = abs(minYpl)
       ypl = np.array([y+aminYpl+1 for y in ypl])
    fig = pyplot.figure()
    ax = pyplot.gca()
    ax.scatter(xpl,ypl)
    ax.set_ylim(min(ypl)-1,max(ypl)+1)
    ax.set_xlim(min(xpl)-1,max(xpl)+1)
    labels = top100List
    for label, x, y in zip(labels, xpl[top100ListIdxs], ypl[top100ListIdxs]):
       pyplot.annotate(label, xy = (x, y), xytext = (-10, 10),textcoords = 'offset points', ha = 'right', va = 'bottom',
           bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.5),
           arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    interactive(True)
    pyplot.show()
    # pyplot.savefig('./images/'+str(year)+'_euclSpaceMapping_via_shortestPaths.jpg', bbox_inches='tight', format='jpg')
    pyplot.savefig('./images/'+str(year)+'_euclSpaceMapping_via_distMatrix.jpg', bbox_inches='tight', format='jpg')
    pyplot.close()

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
top100List = []
with open('./data/top100terms.txt','r') as f:
    next(f)
    for line in f:
        top100List.append(line.split('\t')[0])
top100ListIdxs = [edgeDict['terms'].index(x) for x in top100List]

#set up SOM
n_columns, n_rows = 50, 30
som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid")
savefig = True
#-----------------
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

    # pgRankedTermsList = sorted(edgeDict[year]['term']['pageRank'], key = edgeDict[year]['term']['pageRank'].get,reverse=True)#[:topTermsNum]#[50:2050]
    # pgRnkData = [edgeDict[year]['term']['pageRank'][x] for x in pgRankedTermsList]

    # '''writing term stats to file'''
    # with open('./data/pruned100/watches_nodes_'+year+'.txt', 'w') as d:
    #     d.write('Term\tDegree\tPR\tNPR\n')
    #     for at in top100List:
    #         tmpline = '\t'.join([at,str(edgeDict[year]['term']['degree'][at]),'{0:.8f}'.format(edgeDict[year]['term']['pageRank'][at]),'{0:.8f}'.format(edgeDict[year]['term']['normPageRank'][at])])
    #         d.write(tmpline+'\n')

    # '''creating directed adjacency mat'''
    # print('creating adjacency matrix')
    # adjMat = gDirected.get_adjacency(attribute='weight')
    # adjMat = np.array(adjMat.data)
    # print('writing directed adjacency matrix to file')
    # with open('./data/pruned100/directedAdjacency_matrix_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(nodes)+'\n')
    #     for s in top100List:
    #         distLine = [str(x) for x in adjMat[nodes.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')
    # del(adjMat)

    # edgeDict[year]['directedAdjMat'] = adjMat

    # '''creating undirected adjacency mat'''
    # print('creating undirected adjacency matrix')
    # unDirected = gDirected.as_undirected(combine_edges='sum')
    # # del(gDirected)
    # undirectedAdjMat = np.array(unDirected.get_adjacency(attribute='weight').data)
    # del(unDirected)

    # print('writing undirected adjacency matrix to file')
    # with open('./data/pruned100/undirectedAdjacency_matrix_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(nodes)+'\n')
    #     for s in top100List:
    #         distLine = [str(x) for x in undirectedAdjMat[nodes.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    # edgeDict[year]['undirectedAdjMat'] = undirectedAdjMat

    # '''symmetrical distance matrix extraction'''
    # print('estimate symmetrical distance matrix')
    # from scipy import spatial
    # distMat = spatial.distance.pdist(undirectedAdjMat, 'euclidean')
    # distMat = spatial.distance.squareform(distMat)
    # del(undirectedAdjMat)

    # euclSpaceMapp(gDirected,distMat,top100List,top100ListIdxs)#create euclidean space mapping images

    # '''Write the distance matrix to a file'''
    # print('writing distance matrix to file')
    # with open('./data/pruned100/distance_matrix_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(nodes)+'\n')
    #     for s in top100List:
    #         distLine = [str(float(x)) for x in distMat[nodes.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    # np.savez_compressed('./data/pruned100/distance_matrix_'+year+'.npz',distMat=distMat[top100ListIdxs])

    # '''estimate the potential -(PR_userN * PR_userN+1)/distance matrix'''
    # normMethod = 'SUM'#normalization method of distance matrix
    # if normMethod == 'MAX':
    #     PRversion = 'normPageRank'
    # elif normMethod == 'SUM':
    #     PRversion = 'pageRank'
    # print('estimate potential')
    # potentMat = np.zeros(distMat.shape)
    # pgrMat = np.zeros(distMat.shape)
    # for n,v in edgeDict[year]['term'][PRversion].items():
    #     potentMat[nodes.index(n)] = v
    #     pgrMat[:,nodes.index(n)] = v
    # potentMat = np.multiply(potentMat,pgrMat)
    # PRprodArray = potentMat.reshape(-1)
    # potentMat = (-potentMat)#*1000)
    # distMatPot = distMat + 1
    # distMatPot = distMatPot/distMatPot.sum()#make sure this complies with the normMethod
    # potentMat = np.divide(potentMat,distMatPot)#e-8)
    # potentMat = np.multiply(potentMat,abs(np.identity(potentMat.shape[0])-1))

    # '''estimate the gravity G*(PR_userN * PR_userN+1)/distance^2 matrix'''
    # print('estimate gravity')
    # gravMat = np.zeros(distMat.shape)
    # # pgrMat = np.zeros(distMat.shape)
    # for n,v in edgeDict[year]['term'][PRversion].items():
    #     gravMat[nodes.index(n)] = v
    # #     pgrMat[:,nodes.index(n)] = v
    # gravMat = np.multiply(gravMat,pgrMat)
    # PRprodArray = gravMat.reshape(-1)
    # distMat2 = np.multiply(distMat,distMat)+1
    # distMat2 = distMat2/distMat2.sum()#make sure this complies with the normMethod
    # gravMat = np.divide(gravMat,distMat2)#e-8)
    # gravMat = np.multiply(gravMat,abs(np.identity(gravMat.shape[0])-1))

    # '''estimate and write the gravitational potential associated with mass distribution'''
    # print('estimate and write the gravitational potential associated with mass distribution')
    # with open('./data/pruned100/mass_distr_grav_potential_'+normMethod+'normed_'+year+'.txt', 'w') as d:
    #     d.write('term\tpotential associated with a mass distribution\n')
    #     for s in top100List:
    #         pmd = 0
    #         for i in nodes:
    #             pmd += -edgeDict[year]['term'][PRversion][i]/distMat[nodes.index(s)][nodes.index(i)]
    #         d.write(s+'\t'+str(float(pmd))+'\n')

    # print('Max potential is %s and min potential is %s' %(potentMat.max(),potentMat.min()))
    # print('Max distance is %s and min distance is %s' %(distMat.max(),distMat.min()))
    # del(distMat)
    # print('writing potential matrix to file')
    # with open('./data/pruned100/potential_matrix_'+normMethod+'normed_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(nodes)+'\n')
    #     for s in top100List:
    #         distLine = [str(float(x)) for x in potentMat[nodes.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    # print('Max grav is %s and min grav is %s' %(gravMat.max(),gravMat.min()))
    # print('writing gravity matrix to file')
    # with open('./data/pruned100/gravity_matrix_'+normMethod+'normed_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(nodes)+'\n')
    #     for s in top100List:
    #         distLine = [str(float(x)) for x in gravMat[nodes.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    # np.savez_compressed('./data/pruned100/gravity_matrix_'+normMethod+'normed_'+year+'.npz',gravMat=gravMat[top100ListIdxs])
    # np.savez_compressed('./data/pruned100/potential_matrix_'+normMethod+'normed_'+year+'.npz',potentMat=potentMat[top100ListIdxs])

    # '''Write GRAVITY pairs to txt file for Gephi'''
    # print('writing GRAVITY gephi file')
    # with open('./data/pruned100/forGephi/gravity_'+normMethod+'normed_' + year +'.csv', 'w') as geph:
    #     geph.write('Source,Target,Weight' + '\n')
    #     for s in top100List:
    #         for i in nodes:
    #             if s is not i:
    #                 towrite = [s,i,str(float(-gravMat[nodes.index(s),nodes.index(i)]))]
    #                 geph.write(','.join(towrite) + '\n')
    # #-------------------

    # '''plotting heatmap of potential matrix'''
    # print('plotting heatmap of potential matrix')
    # import seaborn as sns
    # sns.set(style="darkgrid")
    # fig, ax = plt.subplots()
    # ax = sns.heatmap([x[top100ListIdxs] for x in potentMat[top100ListIdxs]])#,xticklabels=2,ax=ax)
    # # ax.set_xticks(range(0, len(top100List), 4))#, minor=False)
    # ax.xaxis.tick_top()
    # ax.set_xticklabels(top100List, minor=False, fontsize = 8, rotation = 90)
    # ax.set_yticklabels(list(reversed(top100List)), minor=False, fontsize = 8)
    # # plt.title('potential matrix using adj matrix')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    # interactive(True)
    # plt.show()
    # fig.savefig('./data/pruned100/images/potentialMat100Heatmap_'+normMethod+'normed_'+year+'.png')
    # plt.close()
    # interactive(False)

    # '''plotting heatmap of gravity matrix'''
    # print('plotting heatmap of gravity matrix')
    # import seaborn as sns
    # sns.set(style="darkgrid")
    # fig, ax = plt.subplots()
    # ax = sns.heatmap([x[top100ListIdxs] for x in gravMat[top100ListIdxs]])#,xticklabels=2,ax=ax)
    # # ax.set_xticks(range(0, len(top100List), 4))#, minor=False)
    # ax.xaxis.tick_top()
    # ax.set_xticklabels(top100List, minor=False, fontsize = 8, rotation = 90)
    # ax.set_yticklabels(list(reversed(top100List)), minor=False, fontsize = 8)
    # # plt.title('gravity matrix using adj matrix')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    # interactive(True)
    # plt.show()
    # fig.savefig('./data/pruned100/images/gravityMat100Heatmap_'+normMethod+'normed_'+year+'.png')
    # plt.close()
    # interactive(False)

    #SOM data extraction from here on--------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    '''Extract Self Organizing Maps of undirected weighted adj mats'''
    df = pd.read_table("data/pruned100/undirectedAdjacency_matrix_" + str(year) + ".txt", sep="\t", header=0,index_col=0)
    dfmax = df.max()
    dfmax[dfmax == 0] = 1
    df = df / dfmax
    labels = df.index.tolist()
    som.update_data(df.values)
    U, s, V = np.linalg.svd(df.values, full_matrices=False)
    if year == yearList[0]:
        epochs = 10
        radius0 = 0
        scale0 = 0.1
    else:
        radius0 = n_rows//5
        scale0 = 0.03
        epochs = 3
    if savefig:
        SOMfilename = "data/pruned100/images/SOM_undirectedAdjacency_matrix_" + str(year) + ".png"
    else:
        SOMfilename = None
    som.train(epochs=epochs, radius0=radius0, scale0=scale0)
    som.view_umatrix(colormap="Spectral_r", bestmatches=True, labels=labels,filename=SOMfilename)
    edgeDict[year]['somCoords'] = [x.tolist() for x in list(som.bmus)]

    '''SOM distance matrix extraction'''
    print('estimate SOM distance matrix')
    from scipy import spatial
    distMat = spatial.distance.pdist(som.bmus, 'euclidean')
    distMat = spatial.distance.squareform(distMat)

    # '''Write the distance matrix to a file'''
    # print('writing distance matrix to file')
    # with open('./data/pruned100/distance_matrix_SOM_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(top100List)+'\n')
    #     for s in top100List:
    #         distLine = [str(float(x)) for x in distMat[top100List.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    # np.savez_compressed('./data/pruned100/distance_matrix_SOM_'+year+'.npz',distMat=distMat)

    '''estimate the potential -(PR_userN * PR_userN+1)/distance matrix'''
    normMethod = 'SUM'#normalization method of distance matrix
    if normMethod == 'MAX':
        PRversion = 'normPageRank'
    elif normMethod == 'SUM':
        PRversion = 'pageRank'
    print('estimate potential')
    potentMat = np.zeros(distMat.shape)
    pgrMat = np.zeros(distMat.shape)
    for n in top100List:
        potentMat[top100List.index(n)] = edgeDict[year]['term'][PRversion][n]
        pgrMat[:,top100List.index(n)] = edgeDict[year]['term'][PRversion][n]
    potentMat = np.multiply(potentMat,pgrMat)
    PRprodArray = potentMat.reshape(-1)
    potentMat = (-potentMat)#*1000)
    distMatPot = distMat + 1
    distMatPot = distMatPot/distMatPot.sum()#make sure this complies with the normMethod
    potentMat = np.divide(potentMat,distMatPot)#e-8)
    potentMat = np.multiply(potentMat,abs(np.identity(potentMat.shape[0])-1))

    '''estimate the gravity G*(PR_userN * PR_userN+1)/distance^2 matrix'''
    print('estimate gravity')
    gravMat = np.zeros(distMat.shape)
    for n in top100List:
        gravMat[top100List.index(n)] = edgeDict[year]['term'][PRversion][n]
    gravMat = np.multiply(gravMat,pgrMat)
    PRprodArray = gravMat.reshape(-1)
    distMat2 = np.multiply(distMat,distMat)+1
    distMat2 = distMat2/distMat2.sum()#make sure this complies with the normMethod
    gravMat = np.divide(gravMat,distMat2)#e-8)
    gravMat = np.multiply(gravMat,abs(np.identity(gravMat.shape[0])-1))

    # '''estimate and write the gravitational potential associated with mass distribution'''
    # print('estimate and write the gravitational potential associated with mass distribution')
    # with open('./data/pruned100/mass_distr_grav_potential_'+normMethod+'normed_'+year+'.txt', 'w') as d:
    #     d.write('term\tpotential associated with a mass distribution\n')
    #     for s in top100List:
    #         pmd = 0
    #         for i in top100List:
    #             pmd += -edgeDict[year]['term'][PRversion][i]/distMat[top100List.index(s)][top100List.index(i)]
    #         d.write(s+'\t'+str(float(pmd))+'\n')

    # print('Max potential is %s and min potential is %s' %(potentMat.max(),potentMat.min()))
    # print('Max distance is %s and min distance is %s' %(distMat.max(),distMat.min()))
    # del(distMat)
    # print('writing potential matrix to file')
    # with open('./data/pruned100/potential_matrix_SOM_'+normMethod+'normed_'+year+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(top100List)+'\n')
    #     for s in top100List:
    #         distLine = [str(float(x)) for x in potentMat[top100List.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    # print('Max grav is %s and min grav is %s' %(gravMat.max(),gravMat.min()))
    # print('writing gravity matrix to file')
    # with open('./data/pruned100/gravity_matrix_SOM_'+normMethod+'normed'+'.txt', 'w') as d:
    #     d.write('Term\t'+'\t'.join(top100List)+'\n')
    #     for s in top100List:
    #         distLine = [str(float(x)) for x in gravMat[top100List.index(s)].tolist()]
    #         d.write(s+'\t'+'\t'.join(distLine)+'\n')

    # np.savez_compressed('./data/pruned100/gravity_matrix_SOM_'+normMethod+'normed_'+year+'.npz',gravMat=gravMat)
    # np.savez_compressed('./data/pruned100/potential_matrix_'+normMethod+'normed_'+year+'.npz',potentMat=potentMat)

    with open('./data/pruned100/sumOfGravity_SOM_'+normMethod+'normed'+'.txt','a') as soG:
        soG.write('Sum of Gravity for year %s is %s\n' %(year,gravMat.sum()/2))

    # '''plotting heatmap of potential matrix using som'''
    # print('plotting heatmap of potential matrix using som')
    # import seaborn as sns
    # sns.set(style="darkgrid")
    # fig, ax = plt.subplots()
    # ax = sns.heatmap(potentMat)#,xticklabels=2,ax=ax)
    # # ax.set_xticks(range(0, len(top100List), 4))#, minor=False)
    # ax.xaxis.tick_top()
    # ax.set_xticklabels(top100List, minor=False, fontsize = 8, rotation = 90)
    # ax.set_yticklabels(list(reversed(top100List)), minor=False, fontsize = 8)
    # # plt.title('potential matrix using SOMs')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    # interactive(True)
    # plt.show()
    # fig.savefig('./data/pruned100/images/SOM_potentialMat100Heatmap_'+normMethod+'normed_'+year+'.png')
    # plt.close()
    # interactive(False)

    # '''plotting heatmap of gravity matrix'''
    # print('plotting heatmap of gravity matrix')
    # import seaborn as sns
    # sns.set(style="darkgrid")
    # fig, ax = plt.subplots()
    # ax = sns.heatmap(gravMat)#,xticklabels=2,ax=ax)
    # # ax.set_xticks(range(0, len(top100List), 4))#, minor=False)
    # ax.xaxis.tick_top()
    # ax.set_xticklabels(top100List, minor=False, fontsize = 8, rotation = 90)
    # ax.set_yticklabels(list(reversed(top100List)), minor=False, fontsize = 8)
    # # plt.title('gravity matrix using SOMs')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    # interactive(True)
    # plt.show()
    # fig.savefig('./data/pruned100/images/SOM_gravityMat100Heatmap_'+normMethod+'normed_'+year+'.png')
    # plt.close()
    # interactive(False)

    # '''plotting kdeplot of PR vs distance'''
    # print('plotting kdeplot of PR vs distance')
    # import seaborn as sns
    # Yarray = gravMat.reshape(-1)
    # PRprodArray = pd.Series(PRprodArray,name='PR product')
    # Yarray = pd.Series(Yarray,name='Euclidean distance')
    # sns.set(style="darkgrid")
    # g = sns.jointplot(PRprodArray, Yarray, kind="kde")
    # g.savefig('./data/pruned100/images/SOM_kdeplot_'+year+'.jpg')

    # '''plot quiver'''
    # X,Y = plt.meshgrid( np.arange(len(top100List)),np.arange(len(top100List)) )
    # plt.figure()
    # Q = plt.quiver( X, Y, gravMat.reshape(-1),  units='width')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    # interactive(True)
    # plt.show()

    # '''Show as image'''
    # plt.figure()
    # im = plt.imshow(gravMat, cmap='hot')
    # plt.colorbar(im, orientation='vertical')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    # interactive(True)
    # plt.show()
    # plt.savefig('./data/pruned100/images/SOM_gravityMat100Image_'+normMethod+'normed_'+year+'.png')
    # plt.close()
    # interactive(False)
    #end of SOM stuff-------------------------------------------------------------------------------------------------

    elapsed = time.time() - t
    print('Elapsed: %.2f seconds' % elapsed)

# '''write pagerank evolution of nodes'''
# print('write pagerank evolution of nodes')
# with open('./data/pruned100/termPageRankEvolution.txt', 'w') as d:
#     d.write('Term\t2006\t2007\t2008\t2009\t2010\t2011\t2012\n')
#     for x in top100List:
#         pgPerYearList = []
#         for y in yearList:
#             pgPerYearList.append(str(float(edgeDict[y]['term']['pageRank'][x])))
#         d.write(x+'\t'+'\t'.join(pgPerYearList)+'\n')


# '''compute velocity, velocity and acceleration of terms based on the undirected adjacency matrices'''
# termKinetics = {x:{'velocity':[],'accel':[]} for x in top100List}
# for s in top100List:
#     for y in yearList[1:]:
#         termKinetics[s]['velocity'].append(distance.euclidean(edgeDict[y]['undirectedAdjMat'][nodes.index(s)],edgeDict[str(int(y)-1)]['undirectedAdjMat'][nodes.index(s)]))
#     termKinetics[s]['accel'] = np.diff(termKinetics[s]['velocity'])
# '''write kinetics of terms'''
# print('write kinetics of terms')
# with open('./data/pruned100/term_displacement.txt', 'w') as d:
#     d.write('Term\t2007\t2008\t2009\t2010\t2011\t2012\n')
#     with open('./data/pruned100/term_velocity.txt', 'w') as v:
#         with open('./data/pruned100/term_acceleration.txt', 'w') as a:
#             for s in top100List:
#                 displLine = [str(float(x)) for x in termKinetics[s]['velocity']]
#                 velLine = [str(float(x)) for x in termKinetics[s]['velocity'].tolist()]
#                 accLine = [str(float(x)) for x in termKinetics[s]['accel'].tolist()]
#                 d.write(s+'\t'+'\t'.join(displLine)+'\n')
#                 v.write(s+'\t'+'\t'.join(velLine)+'\n')
#                 a.write(s+'\t'+'\t'.join(accLine)+'\n')

'''compute velocity and acceleration of terms based on the Self-Organizing Maps of the undirected adjacency matrices'''
termKinetics = {x:{'velocity':[],'accel':[], 'force':[], 'kinEnergy':[]} for x in top100List}

for idx,s in enumerate(top100List):
    for y in yearList[1:]:
        veloc = distance.euclidean(edgeDict[y]['somCoords'][idx],edgeDict[str(int(y)-1)]['somCoords'][idx])
        if y == yearList[1]:
            termKinetics[s]['velocity'].append(veloc)
            termKinetics[s]['kinEnergy'].append(0.5*edgeDict[str(int(y)-1)]['term']['pageRank'][s]*(veloc**2))
        termKinetics[s]['velocity'].append(veloc)
        termKinetics[s]['kinEnergy'].append(0.5*edgeDict[y]['term']['pageRank'][s]*(veloc**2))
    accel = np.diff(termKinetics[s]['velocity'])
    accel[0] = accel[1]
    termKinetics[s]['accel'] = np.append(accel[0],accel).tolist()
    for idy,y in enumerate(yearList):
        termKinetics[s]['force'].append(edgeDict[y]['term']['pageRank'][s]*termKinetics[s]['accel'][idy])

with open('./data/pruned100/sumOfKinEner_SOM_'+normMethod+'normed_'+year+'.txt','w') as soKE:
    for idy,y in enumerate(yearList):
        keSum = []
        for s in top100List:
            keSum.append(termKinetics[s]['kinEnergy'][idy])
        soKE.write('Sum of Kinetic Energy for year %s is %s\n' %(y,str(float(sum(keSum)))))



# '''write kinetics of terms'''
# print('write kinetics of terms')
# with open('./data/pruned100/SOM_term_kineticEnergy.txt', 'w') as ke:
#     ke.write('Term\t2006\t2007\t2008\t2009\t2010\t2011\t2012\n')
#     with open('./data/pruned100/SOM_term_velocity.txt', 'w') as v:
#         v.write('Term\t2006\t2007\t2008\t2009\t2010\t2011\t2012\n')
#         with open('./data/pruned100/SOM_term_acceleration.txt', 'w') as a:
#             a.write('Term\t2006\t2007\t2008\t2009\t2010\t2011\t2012\n')
#             with open('./data/pruned100/SOM_term_Force.txt', 'w') as f:
#                 f.write('Term\t2006\t2007\t2008\t2009\t2010\t2011\t2012\n')
#                 for s in top100List:
#                     velLine = [str(float(x)) for x in termKinetics[s]['velocity']]
#                     accLine = [str(float(x)) for x in termKinetics[s]['accel']]
#                     kEnerLine = [str(float(x)) for x in termKinetics[s]['kinEnergy']]
#                     forceLine = [str(float(x)) for x in termKinetics[s]['force']]
#                     v.write(s+'\t'+'\t'.join(velLine)+'\n')
#                     a.write(s+'\t'+'\t'.join(accLine)+'\n')
#                     ke.write(s+'\t'+'\t'.join(kEnerLine)+'\n')
#                     f.write(s+'\t'+'\t'.join(forceLine)+'\n')


pickle.dump(edgeDict,open('./data/pruned/pickles/edgeDictPruned.pck','wb'), protocol = 2)
elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
