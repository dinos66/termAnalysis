# -*- coding: utf-8 -*-
'''
    Create adjacency matrices and analyse terms dynamically
'''
print('Create dynamic adjacency matrices and ESOMs')
#--------------------------------------------
#run create_Info_Files.py before running this
#--------------------------------------------
import pickle, time, igraph, glob, os, somoclu, collections
import itertools, codecs, seaborn, math, pprint, random, re
from matplotlib import rc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from scipy.spatial import distance
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.colors as colors
import seaborn as sns
import sklearn.cluster as clusterAlgs
#--------------------------------------------
print(time.asctime( time.localtime(time.time()) ))
t = time.time()

edgeReadPath = './data/artworks_edges/dynamic'
adjMatWritePath = './data/artworks_adjacencyMats/dynamic'
distMatWritePath = './data/artworks_distanceMats/dynamic'
potentMatWritePath = './data/artworks_potentialMats/dynamic'
gravMatWritePath = './data/artworks_gravityMats/dynamic'
umatrixWritePath = './data/artworks_UMX/dynamic'
figWritePath = './data/artworks_figs/dynamic'
greenwichFigWritePath = figWritePath+'/greenwich'
greenwichUmatrixWritePath = umatrixWritePath+'/greenwich'
gephiWritePath = './data/artworks_gephi/dynamic'
statsWritePath = './data/artworks_stats'
if not os.path.exists('./data/artworks_tmp'):
    os.makedirs('./data/artworks_tmp')
if not os.path.exists(adjMatWritePath):
    os.makedirs(adjMatWritePath)
if not os.path.exists(distMatWritePath):
    os.makedirs(distMatWritePath)
if not os.path.exists(potentMatWritePath):
    os.makedirs(potentMatWritePath)
    os.makedirs(gravMatWritePath)
if not os.path.exists(umatrixWritePath):
    os.makedirs(umatrixWritePath)
if not os.path.exists(figWritePath):
    os.makedirs(figWritePath)
if not os.path.exists(gephiWritePath):
    os.makedirs(gephiWritePath)
if not os.path.exists(greenwichFigWritePath):
    os.makedirs(greenwichFigWritePath)
if not os.path.exists(greenwichUmatrixWritePath):
    os.makedirs(greenwichUmatrixWritePath)


LVLs = ['lvlA']#['lvl1','lvl2','lvl3','lvlA'] #'lvl1','lvl2','lvl3',
heatmapFonts = [4]#[12,7,6,4]#12,7,6,
yearPeriods = ['2000s'] #['1800s','2000s'] 
trueYearsIni = [1964]#[1800,1964]

termLabelDict = pickle.load(open('./data/artworks_verification_labels/WmatrixLabelDict.pck','rb'))

def recRank(mylist):#Perform the Reciprocal Rank Fusion for a list of rank values
    finscore = []
    mylist=[x+1 for x in mylist]
    for rank in mylist:
        finscore.append(1/(20+rank))
    return sum(finscore)

def toroidDistance(myarray,width,height):
    somDim2 = []
    for idx,x in enumerate(myarray[:-1]):
        newxa = myarray[idx+1:]
        for nx in newxa:
            somDim2.append(np.sqrt(min(abs(x[0] - nx[0]), width - abs(x[0] - nx[0]))**2 + min(abs(x[1] - nx[1]), height - abs(x[1]-nx[1]))**2))
    SD = np.array(somDim2)
    return distance.squareform(SD)

def toroidDistanceSingle(coords1,coords2,width,height):
    return (np.sqrt(min(abs(coords1[0] - coords2[0]), width - abs(coords1[0] - coords2[0]))**2 + min(abs(coords1[1] - coords2[1]), height - abs(coords1[1]-coords2[1]))**2))

def toroidCoordinateFinder(coorx,distx,coory,disty,w,h):
    if coorx+distx>=w:
        ncx = coorx+distx-w
    elif coorx+distx<0:
        ncx = w+coorx+distx
    else:
        ncx = coorx+distx
    if coory+disty>=h:
        ncy = coory+disty-h
    elif coory+disty<0:
        ncy = h+coory+disty
    else:
        ncy = coory+disty
    return (ncx,ncy)

for lIdx,lvl in enumerate(LVLs):
    heatMapFont = heatmapFonts[lIdx]
    for idy,years in enumerate(yearPeriods):
        files = glob.glob(edgeReadPath+'/'+years+lvl+'_*.csv')
        files.sort(key=lambda x: os.path.getmtime(x))
        try:
            edgeDict = pickle.load(open('./data/artworks_tmp/edgeDictDynamic'+years+lvl+'.pck','rb'))
        except:
            edgeDict = {'uniquePersistentTerms':[]}
            termsYears = []
            for filename in files:
                periodIdx = filename[filename.index(lvl)+5:-4]
                tmpTerms = []
                edgeDict[periodIdx] = {}
                with codecs.open(filename, 'r','utf8') as f:
                    # print(filename)
                    adjList = []
                    next(f)
                    for line in f:
                        line = line.split(',')
                        tripletuple = line[:2]
                        tmpTerms.extend(tripletuple)
                        tripletuple.append(int(line[2].strip()))
                        adjList.append(tuple(tripletuple))
                    edgeDict[periodIdx]['adjList'] = adjList
                termsYears.append(list(set(tmpTerms)))
                print('There are %s unique nodes for period %s' %(len(termsYears[-1]),periodIdx))

            repetitiveTerms = collections.Counter(list(itertools.chain.from_iterable(termsYears)))
            edgeDict['allTerms'] = list(repetitiveTerms.keys())
            edgeDict['uniquePersistentTerms'] = [x for x,v in repetitiveTerms.items() if v == len(files)]
            edgeDict['uniquePersistentTerms'].sort()
            pass

        with open(statsWritePath+'/'+years+lvl+'_unique_persistent_terms.txt','w') as f:
            for word in edgeDict['uniquePersistentTerms']:
                f.write(word+'\n')

        statement = ('For %s in the %s there are %s unique persistent terms globally out of %s unique terms' %(lvl,years,len(edgeDict['uniquePersistentTerms']),len(edgeDict['allTerms'])))
        time.sleep(5)
        print(statement)
        
        '''set up SOM'''#--------------------------------------------------------------------
##        n_columns, n_rows = 200, 120 
##        lablshift = 1
        if lvl == 'lvl1':
           n_columns, n_rows = 20, 12
           lablshift = .2
        elif lvl == 'lvl2':
           n_columns, n_rows = 40, 24
           lablshift = .3
        elif lvl == 'lvl3':
           n_columns, n_rows = 50, 30
           lablshift = .4
        elif lvl == 'lvlA':
           n_columns, n_rows = 60, 40
           lablshift = .5
        epochs2 = 3
        som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid", initialization="pca")
        savefig = True
        SOMdimensionsString = 'x'.join([str(x) for x in [n_columns,n_rows]])
        #--------------------------------------------------------------------------------
        yearList = []
        count = 0
        termPrRanks, termAuthRanks, termHubRanks, termBetweenRanks = {}, {}, {}, {}
        for filename in files:
            periodIdx = filename[filename.index(lvl)+5:-4]
##            if periodIdx != '7':
##                continue
            yearList.append(periodIdx)
            print(periodIdx)
            # try:
            #     gUndirected = edgeDict[periodIdx]['graph']
            # except:
            gUndirected=igraph.Graph.Full(0, directed = False)
            gUndirected.es['weight'] = 1
            '''ReRanking the nodes based on their reciprocal rank between timeslots'''
            try:
                gUndirected.add_vertices(edgeDict['topTermsByPR'])
                print('used top Terms By PageRank')
                # print(edgeDict['topTermsByPR'][:5])
            except:
                gUndirected.add_vertices(edgeDict['uniquePersistentTerms'])
                print('used alphabetically ranked terms')
                pass            
            myEdges,myWeights = [], []
            nodesWithEdges = []
            for x in edgeDict[periodIdx]['adjList']:
                if x[0] in edgeDict['uniquePersistentTerms'] and x[1] in edgeDict['uniquePersistentTerms']:
                    myEdges.append((x[0],x[1]))
                    myWeights.append(x[2])
                    nodesWithEdges.extend(x[:2])
            print('Full No of edges: %s and pruned No of edges %s' %(len(edgeDict[periodIdx]['adjList']),len(myEdges)))
            gUndirected.add_edges(myEdges)
            gUndirected.es["weight"] = myWeights
            edgeDict[periodIdx]['graph'] = gUndirected
            gUndirected.vs['label'] = gUndirected.vs['name']

            nodes = gUndirected.vs['name']
            # print(nodes[:5])

            #--------------------------------------------------------------------------------
            '''Extract centrality measures'''#-----------------------------------------------
            #--------------------------------------------------------------------------------
            edgeDict[periodIdx]['term'] = {'degree':{},'pageRank':{},'maxnormPageRank':{}, 'minnormPageRank':{}, 'authority':{}, 'hub':{}, 'betweenness':{}}
            pageRank = gUndirected.pagerank(weights = 'weight', directed=False)
            authority = gUndirected.authority_score(weights = 'weight') #HITS authority score
            hub = gUndirected.hub_score(weights = 'weight')#HITS hub score
            betweenness = gUndirected.betweenness(weights = 'weight', directed = False)
            # print('extracted pagerank')
            maxPR = max(pageRank)
            maxnormPageRank = [x/maxPR for x in pageRank]
            minPR = min(pageRank)
            minnormPageRank = [x/minPR for x in pageRank]
            maxminPr = max(minnormPageRank)
            minmaxPRdiff = maxPR-minPR
            minmaxnormPageRank = [1+3*((x-minPR)/minmaxPRdiff) for x in pageRank]
            for x in nodes:
                edgeDict[periodIdx]['term']['pageRank'][x] = pageRank[nodes.index(x)]
                edgeDict[periodIdx]['term']['maxnormPageRank'][x] = maxnormPageRank[nodes.index(x)]
                edgeDict[periodIdx]['term']['minnormPageRank'][x] = minnormPageRank[nodes.index(x)]
                edgeDict[periodIdx]['term']['degree'][x] = gUndirected.degree(x)
                edgeDict[periodIdx]['term']['authority'][x] = authority[nodes.index(x)]
                edgeDict[periodIdx]['term']['hub'][x] = hub[nodes.index(x)]
                edgeDict[periodIdx]['term']['betweenness'][x] = betweenness[nodes.index(x)]
            tmpPRrank = sorted(edgeDict[periodIdx]['term']['pageRank'], key=lambda k: [edgeDict[periodIdx]['term']['pageRank'][k],edgeDict[periodIdx]['term']['degree'][k],k],reverse =True)
            for x in nodes:
                if x not in termPrRanks:
                    termPrRanks[x] = [tmpPRrank.index(x)]
                else:
                    termPrRanks[x].append(tmpPRrank.index(x))

            tmpAuthrank = sorted(edgeDict[periodIdx]['term']['authority'], key=lambda k: [edgeDict[periodIdx]['term']['authority'][k],edgeDict[periodIdx]['term']['degree'][k],k],reverse =True)
            for x in nodes:
                if x not in termAuthRanks:
                    termAuthRanks[x] = [tmpAuthrank.index(x)]
                else:
                    termAuthRanks[x].append(tmpAuthrank.index(x))

            tmpHubrank = sorted(edgeDict[periodIdx]['term']['hub'], key=lambda k: [edgeDict[periodIdx]['term']['hub'][k],edgeDict[periodIdx]['term']['degree'][k],k],reverse =True)
            for x in nodes:
                if x not in termHubRanks:
                    termHubRanks[x] = [tmpHubrank.index(x)]
                else:
                    termHubRanks[x].append(tmpHubrank.index(x))

            tmpBetweenrank = sorted(edgeDict[periodIdx]['term']['betweenness'], key=lambda k: [edgeDict[periodIdx]['term']['betweenness'][k],edgeDict[periodIdx]['term']['degree'][k],k],reverse =True)
            for x in nodes:
                if x not in termBetweenRanks:
                    termBetweenRanks[x] = [tmpBetweenrank.index(x)]
                else:
                    termBetweenRanks[x].append(tmpBetweenrank.index(x))
            # -----------------------------------------------------------------------------------------------

            #-----------------------------------------------------------------------------------------------
            '''creating undirected adjacency mat'''#--------------------------------------------------------
            #-----------------------------------------------------------------------------------------------
            if not os.path.exists(adjMatWritePath):
                os.makedirs(adjMatWritePath)

            print('creating adjacency matrix')
            adjMat = gUndirected.get_adjacency(attribute='weight')
            adjMat = np.array(adjMat.data)

            print('writing undirected adjacency matrix to file')
            with open(adjMatWritePath+'/AdjMat'+years+lvl+'_'+periodIdx+'.txt', 'w') as d:
                d.write('Term\t'+'\t'.join(nodes)+'\n')
                for s in nodes:
                    distLine = [str(x) for x in adjMat[nodes.index(s)].tolist()]
                    d.write(s+'\t'+'\t'.join(distLine)+'\n')

            #-----------------------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------------------------------------------------
            '''SOM data extraction from here on------------------------------------------------------------------------------------'''
            # ------------------------------------------------------------------------------------------------------------------------
            '''Extract Self Organizing Maps of undirected weighted adj mats'''#change filename depending on labeled or numbered terms
            nummedOrNot = ''#'nummed' are the labels numbers or text (leave blank)?
            labelFormat = 'code' #switch terms by Wmatrix code or label?
            df = pd.read_table(adjMatWritePath+'/'+nummedOrNot+'AdjMat'+years+lvl+'_'+periodIdx+'.txt', sep="\t", header=0,index_col=0)
            dfmax = df.max()
            dfmax[dfmax == 0] = 1
            df = df / dfmax
            originallabels = df.index.tolist()
            # print(originallabels[:5])
            labels = originallabels # labels = [termLabelDict[nodes[x]][labelFormat] for x in originallabels] #switch terms by Wmatrix code or label?
            som.update_data(df.values)
            U, s, V = np.linalg.svd(df.values, full_matrices=False)
            if periodIdx == yearList[0]:
                epochs = 10
                radius0 = 0
                scale0 = 0.1
            else:
                radius0 = n_rows//5
                scale0 = 0.03
                epochs = epochs2

            #-------clustering params---------------            
            # algorithm = clusterAlgs.SpectralClustering()
            clusterAlgLabel = 'KMeans8'# KMeans8 , SpectralClustering
            #---------------------------------------
            if savefig:
                if not os.path.exists(figWritePath+'/Clusters/'+clusterAlgLabel+'/SOMs/'+SOMdimensionsString+'_epochs'+str(epochs2)):
                    os.makedirs(figWritePath+'/Clusters/'+clusterAlgLabel+'/SOMs/'+SOMdimensionsString+'_epochs'+str(epochs2))
                SOMfilename = figWritePath+'/Clusters/'+clusterAlgLabel+'/SOMs/'+SOMdimensionsString+'_epochs'+str(epochs2)+'/SOM_'+nummedOrNot+'AdjMat'+years+lvl+'_'+periodIdx+'.png'
                SOMfilenameNoLabels = figWritePath+'/Clusters/'+clusterAlgLabel+'/SOMs/'+SOMdimensionsString+'_epochs'+str(epochs2)+'/noLabelsSOM_AdjMat'+years+lvl+'_'+periodIdx+'.png'                
                # SOMfilenameNoBMUs = figWritePath+'/Clusters/'+clusterAlgLabel+'/SOMs/'+SOMdimensionsString+'_epochs'+str(epochs2)+'/noBMUsSOM_AdjMat'+years+lvl+'_'+periodIdx+'.png'
            else:
                SOMfilename = None
            som.train(epochs=epochs, radius0=radius0, scale0=scale0)
            #----------------------clustering-----------
            try:
                som.cluster(algorithm=algorithm)
                print('Clustering algorithm employed: %s' %clusterAlgLabel)
            except:
                som.cluster()
                print('Clustering algorithm employed: K-means with 8 centroids')
                pass
            #----------------------clustering-----------
            rc('font', **{'size': 11}); figsize = (20, 20/float(n_columns/n_rows))
            som.view_umatrix(figsize = figsize, colormap="Spectral_r", bestmatches=True, labels=labels,filename=SOMfilename)
            plt.close()
            som.view_umatrix(figsize = figsize, colormap="Spectral_r", bestmatches=True, filename=SOMfilenameNoLabels)
            plt.close()
            # som.view_umatrix(figsize = figsize, colormap="Spectral_r", filename=SOMfilenameNoBMUs)
            # plt.close()
            edgeDict[periodIdx]['somCoords'] = {SOMdimensionsString:som.bmus}

            colors = []
            for bm in som.bmus:
                colors.append(som.clusters[bm[1], bm[0]])
            # areas = [200]*len(som.bmus)
            areas = [x*70 for x in minmaxnormPageRank]
            #-----------------------------------------------------------------------------------------------
            
            #-----------------------------------------------------------------------------------------------
            '''write and show the umatrix (umx)'''#---------------------------------------------------------
            #-----------------------------------------------------------------------------------------------
##            somUmatrix =  edgeDict[periodIdx]['somUmatrix'][SOMdimensionsString]
##            print('writing umatrix to file')
##            np.savetxt(umatrixWritePath+'/umx'+years+lvl+'_'+periodIdx+'.umx',somUmatrix,delimiter='\t', newline='\n',header='% '+ '%s %s'%(n_rows,n_columns))
##
##            print('writing BMU coords to file')
##            with open(umatrixWritePath+'/umx'+years+lvl+'_'+periodIdx+'.bm','w') as f:
##                with open(umatrixWritePath+'/umx'+years+lvl+'_'+periodIdx+'.names','w') as fn:
##                    f.write('% '+'%s %s\n' %(n_rows,n_columns))
##                    fn.write('% '+str(len(nodes))+'\n')
##                    for idx,coos in enumerate(edgeDict[periodIdx]['somCoords'][SOMdimensionsString]):
##                        f.write('%s %s %s\n' %(idx,coos[1],coos[0]))
##                        fn.write('%s %s %s\n' %(idx,nodes[idx],nodes[idx]))
##
##            print('plotting umatrix 3D surface') 
##            fig = plt.figure()
##            ax = fig.gca(projection='3d')
##            X = np.arange(0, n_columns, 1)
##            Y = np.arange(0, n_rows, 1)
##            X, Y = np.meshgrid(X, Y)
##            N=somUmatrix/somUmatrix.max()
##            surf = ax.plot_surface(X, Y, somUmatrix, facecolors=cm.jet(N),rstride=1, cstride=1)#,facecolors=cm.jet(somUmatrix) cmap=cm.coolwarm, linewidth=0, antialiased=False)
##            m = cm.ScalarMappable(cmap=cm.jet)
##            m.set_array(somUmatrix)
##            plt.colorbar(m, shrink=0.5, aspect=5)
##            plt.title('SOM umatrix 3D surface vizualization (Level '+lvl+' terms | 5 year period prior to '+str(int(periodIdx)*5+trueYearsIni[idy])+')')
##            mng = plt.get_current_fig_manager()
##            mng.window.state('zoomed')
##            interactive(True)
##            plt.show()            
##            fig.savefig(figWritePath+'/SOM Umatrices/umxSurf'+years+lvl+'_'+periodIdx+'.png',bbox_inches='tight')
##            plt.close()
##            interactive(False)

            #-----------------------------------------------------------------------------------------------
            '''Plotting BMU coordinates with labels'''#-----------------------------------------------------
            #-----------------------------------------------------------------------------------------------
##            labelFormat = 'code'
##            fig, ax = plt.subplots()
##            xDimension = [x[0] for x in edgeDict[periodIdx]['somCoords'][SOMdimensionsString]]#[:10]]
##            yDimension = [x[1] for x in edgeDict[periodIdx]['somCoords'][SOMdimensionsString]]#[:10]] 
##            plt.scatter(xDimension,yDimension, c=colors, s = areas, alpha = 0.7)
##            labels = [str(colors[x])+'_'+termLabelDict[" ".join(re.findall("[a-zA-Z]+", nodes[x]))][labelFormat] for x in range(len(xDimension))]
##            doneLabs = set([''])
##            for label, x, y in zip(labels, xDimension, yDimension):
##                lblshiftRatio = 2
##                labFinshift = ''
##                while labFinshift in doneLabs:
##                    potentialPositions = [(x, y+lablshift), (x+lblshiftRatio*lablshift, y), (x-lblshiftRatio*lablshift, y), (x+lblshiftRatio*lablshift, y+lblshiftRatio*lablshift), 
##                    (x-lblshiftRatio*lablshift, y+lblshiftRatio*lablshift), (x+lblshiftRatio*lablshift, y-lblshiftRatio*lablshift), (x+lblshiftRatio*lablshift, y+lblshiftRatio*lablshift),
##                    (x-lblshiftRatio*lablshift, y+lblshiftRatio*lablshift)]
##                    for pP in potentialPositions:
##                        labFinshift = pP
##                        if labFinshift not in doneLabs:
##                            break
##                    lblshiftRatio+=1
##                doneLabs.add(labFinshift)
##                plt.annotate(label, xy = (x, y), xytext = labFinshift, textcoords = 'data', ha = 'center', va = 'center',bbox = dict(boxstyle = 'round,pad=0.1', fc = 'white', alpha = 0.4))
##                lIdx+=1
##
##            xCc = [x[1] for x in som.centroidBMcoords]
##            yCc = [x[0] for x in som.centroidBMcoords]
##            plt.scatter(xCc,yCc, c= range(len(som.centroidBMcoords)), s= [1000]*len(som.centroidBMcoords), alpha = 0.4)
##
##            plt.xlim(0,n_columns)
##            plt.ylim(0,n_rows) 
##            ax.invert_yaxis() 
##            plt.title('Labeled SOM. Level '+lvl+' terms, timeslot '+periodIdx+' (5 year period prior to '+str(int(periodIdx)*5+trueYearsIni[idy])+')')
##            mng = plt.get_current_fig_manager()
##            mng.window.state('zoomed')
##            interactive(True)
##            plt.show()            
##            fig.savefig(figWritePath+'/Clusters/'+clusterAlgLabel+'/SOMs/'+SOMdimensionsString+'_epochs'+str(epochs2)+'/SOM_Wmatrix'+labelFormat+'LabeledAdjMat'+years+lvl+'_'+periodIdx+'.png',bbox_inches='tight')
##            plt.close()
##            interactive(False)
            #-----------------------------------------------------------------------------------------------

        '''pageRank and HITS term fluctuation'''
        numOfPlots = [5, 10, 20]
        marker, color = ['*', '+', 'o','d','h','p','s','v','^','d'], ['g','r','m','c','y','k']#line, ["-","--","-.",":"] #list(colors.cnames.keys())
        marker.sort()
        color.sort()
        asmarker = itertools.cycle(marker) 
        ascolor = itertools.cycle(color) 
        # asline = itertools.cycle(line) 

        if not os.path.exists(figWritePath+'/centrality fluctuations over time/PageRank'):
            os.makedirs(figWritePath+'/centrality fluctuations over time/PageRank')
            os.makedirs(figWritePath+'/centrality fluctuations over time/HITS')
            os.makedirs(figWritePath+'/centrality fluctuations over time/Betweenness')

        allPeriods = list(edgeDict.keys())
        allPeriods.remove('uniquePersistentTerms')
        allPeriods.remove('allTerms')            
        try:
            allPeriods.remove('topTermsByPR')
        except:
            pass
        allPeriods.sort()

        termPRRankDict = {}
        termPRSequences = {}
        termAuthRankDict = {}
        termAuthSequences = {}
        termHubRankDict = {}
        termHubSequences = {}
        termBetweenRankDict = {}
        termBetweenSequences = {}
        for x in nodes:
            prSequence, authSequence, hubSequence, betweenSequence = [], [] ,[], []
            for p in allPeriods:
                prSequence.append(edgeDict[p]['term']['pageRank'][x])
                authSequence.append(edgeDict[p]['term']['authority'][x])
                hubSequence.append(edgeDict[p]['term']['hub'][x])
                betweenSequence.append(edgeDict[p]['term']['betweenness'][x]) 
            termPRSequences[x] = prSequence
            termPRRankDict[x] = recRank(termPrRanks[x])
            termAuthSequences[x] = authSequence
            termAuthRankDict[x] = recRank(termAuthRanks[x])
            termHubSequences[x] = hubSequence
            termHubRankDict[x] = recRank(termHubRanks[x])
            termBetweenSequences[x] = betweenSequence
            termBetweenRankDict[x] = recRank(termBetweenRanks[x])
        termPRRanked = sorted(termPRRankDict, key=termPRRankDict.get, reverse=True)
        termAuthRanked = sorted(termAuthRankDict, key=termAuthRankDict.get, reverse=True)
        termHubRanked = sorted(termHubRankDict, key=termHubRankDict.get, reverse=True)
        termBetweenRanked = sorted(termBetweenRankDict, key=termBetweenRankDict.get, reverse=True)

        edgeDict['topTermsByPR'] = termPRRanked

        pickle.dump(edgeDict,open('./data/artworks_tmp/edgeDictDynamic'+years+lvl+'.pck','wb'), protocol = 2)

        

elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
