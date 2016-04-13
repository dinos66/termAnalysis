'''
	Create adjacency matrices and analyse terms
'''
print('Create adjacency matrices and ESOMs')
#--------------------------------------------
#run create_Info_Files.py before running this
#--------------------------------------------
import pickle, time, igraph, glob, os, somoclu
import pandas as pd
import numpy as np
#--------------------------------------------
print(time.asctime( time.localtime(time.time()) ))
t = time.time()

edgeReadPath = './data/artworks_edges'
lvl = 'lvl2'
adjMatWritePath = './data/artworks_adjacencyMats'
figWritePath = './data/artworks_figs'
yearPeriods = ['1800s','2000s']
for years in yearPeriods:

    try:
        edgeDict = pickle.load(open('./data/artworks_tmp/edgeDictStatic'+years+lvl+'.pck','rb'))
    except:
        edgeDict = {'uniqueTerms':[]}
        with open(edgeReadPath+'/staticEdges'+years+lvl+'.csv', 'r') as f:
            adjList = []
            next(f)
            for line in f:
                line = line.split(',')
                tripletuple = line[:2]
                edgeDict['uniqueTerms'].extend(tripletuple)
                tripletuple.append(int(line[2].strip()))
                adjList.append(tuple(tripletuple))
        edgeDict['adjList'] = adjList
        edgeDict['uniqueTerms'] = list(set(edgeDict['uniqueTerms']))
        edgeDict['uniqueTerms'].sort()


    try:
        gUndirected = edgeDict['graph']
    except:
        gUndirected=igraph.Graph.TupleList(edgeDict['adjList'], directed = False, weights=True)
        edgeDict['graph'] = gUndirected
        print('created graph')

    nodes = gUndirected.vs['name']
    gUndirected.vs['label'] = gUndirected.vs['name']

    edgeDict['term'] = {'degree':{},'pageRank':{},'normPageRank':{}}
    pageRank = gUndirected.pagerank(weights = 'weight')
    print('extracted pagerank')
    maxPR = max(pageRank)
    normPageRank = [x/maxPR for x in pageRank]
    for x in nodes:
        edgeDict['term']['pageRank'][x] = pageRank[nodes.index(x)]
        edgeDict['term']['normPageRank'][x] = normPageRank[nodes.index(x)]
        edgeDict['term']['degree'][x] = gUndirected.degree(x)

    '''creating directed adjacency mat--------------------------------------------------------'''
    if not os.path.exists(adjMatWritePath):
        os.makedirs(adjMatWritePath)
        
    print('creating adjacency matrix')
    adjMat = gUndirected.get_adjacency(attribute='weight')
    adjMat = np.array(adjMat.data)
    print('writing directed adjacency matrix to file')

    with open(adjMatWritePath+'/undirectedAdjacency_matrix'+years+lvl+'.txt', 'w') as d:
        d.write('Term\t'+'\t'.join(nodes)+'\n')
        for s in nodes:
            distLine = [str(x) for x in adjMat[nodes.index(s)].tolist()]
            d.write(s+'\t'+'\t'.join(distLine)+'\n')
    '''---------------------------------------------------------------------------------------'''
    #SOM data extraction from here on--------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    '''Extract Self Organizing Maps of undirected weighted adj mats'''
    if not os.path.exists(figWritePath):
        os.makedirs(figWritePath)
    n_columns, n_rows = 160, 100
    som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid")
    savefig = True

    df = pd.read_table(adjMatWritePath+'/undirectedAdjacency_matrix'+years+lvl+'.txt', sep="\t", header=0,index_col=0)
    dfmax = df.max()
    dfmax[dfmax == 0] = 1
    df = df / dfmax
    labels = df.index.tolist()
    som.update_data(df.values)
    U, s, V = np.linalg.svd(df.values, full_matrices=False)
    epochs, radius0, scale0 = 10, 0, 0.2
    if savefig:
        SOMfilename = figWritePath+'/SOM_undirectedAdjacency_matrix'+years+lvl+'.png'
    else:
        SOMfilename = None
    som.train(epochs=epochs, radius0=radius0, scale0=scale0)
    som.view_umatrix(colormap="Spectral_r", bestmatches=True, labels=labels,filename=SOMfilename)#, figsize = (800,600))
    edgeDict['somCoords'] = [x.tolist() for x in list(som.bmus)]


    if not os.path.exists('./data/artworks_tmp'):
        os.makedirs('./data/artworks_tmp')
    pickle.dump(edgeDict,open('./data/artworks_tmp/edgeDictStatic'+years+lvl+'.pck','wb'), protocol = 2)

elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
