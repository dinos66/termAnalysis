# -*- coding: utf-8 -*-
'''
    Create adjacency matrices and analyse terms dynamically
'''
print('Create document UCREL ESOMs')
#--------------------------------------------
#run create_Info_Files.py before running this
#--------------------------------------------
import pickle, time, igraph, glob, os, somoclu, collections
import itertools, codecs, seaborn, math, pprint, random
from matplotlib import rc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
import matplotlib.colors as colors
from scipy.spatial import distance
import seaborn as sns
import sklearn.cluster as clusterAlgs
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as sch
#--------------------------------------------
print(time.asctime( time.localtime(time.time()) ))
t = time.time()

'''Select ucrel format to use: label or code'''
UCRELformats = ['label'] # code label


readDataPath = './data/artworks_tmp/artworks_doc_UCREL'
statsWritePath = './data/artworks_stats'
figWritePath = './data/artworks_figs/doc_UCRELfigs'
splitsMergesWritePath = './data/splitsMerges/doc_UCREL'
if not os.path.exists(figWritePath):
    os.makedirs(figWritePath)

LVLs = ['lvl1','lvl2','lvl3','lvlA'] #'lvl1','lvl2','lvl3',
yearPeriods = ['1800s','2000s'] #['2000s']#'1800s',
trueYearsIni = [1800,1964]#[1964]# 1800,
leaffont = [14,11,7,4]#12,7,6,

for UCRELformat in UCRELformats:
    print('format is %s' %UCRELformat)
    for idy,years in enumerate(yearPeriods):
        print(years)    
        for lIdx,lvl in enumerate(LVLs):
            print(lvl)
            persistUCREL = [x.strip() for x in open(statsWritePath+'/'+years+lvl+'_unique_persistent_UCREL'+UCRELformat+'.txt','r').readlines()]
            dataDict = {'uniquePersistentUCRELs':persistUCREL}
            for pIdx in range(10):
                periodIdx = str(pIdx)
                dataDict[periodIdx] = {}
                '''set up SOM'''#--------------------------------------------------------------------
                # n_columns, n_rows = 100, 60 
                # n_columns, n_rows = 150, 90
                n_columns, n_rows = 200, 120
                lablshift = 1 
                #------------------------
                # if lvl == 'lvl1':
                #  n_columns, n_rows = 20, 12
                #  lablshift = .2
                # elif lvl == 'lvl2':
                #  n_columns, n_rows = 40, 24
                #  lablshift = .3
                # elif lvl == 'lvl3':
                #  n_columns, n_rows = 50, 30
                #  lablshift = .4
                # elif lvl == 'lvlA':
                #  n_columns, n_rows = 60, 40
                #  lablshift = .5 #------------
                som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid", initialization="pca")
                savefig = True
                SOMdimensionsString = 'x'.join([str(x) for x in [n_columns,n_rows]])
                if pIdx<1:
                    print('SOM dimension is: %s' %SOMdimensionsString)
                # #--------------------------------------------------------------------------------

                # ------------------------------------------------------------------------------------------------------------------------
                '''SOM data extraction from here on------------------------------------------------------------------------------------'''
                # ------------------------------------------------------------------------------------------------------------------------
                '''Extract Self Organizing Maps of undirected weighted adj mats'''
                df = pd.read_table(readDataPath+'/'+UCRELformat+'OccurrenceMat'+years+lvl+'_'+periodIdx+'.tsv', sep="\t", header=0,index_col=0)
                df = df.transpose()
                dfmax = df.max()
                dfmax[dfmax == 0] = 1
                df = df / dfmax
                labels = df.index.tolist()
                nodes = df.index.tolist()
                som.update_data(df.values)
                epochs = 10
                radius0 = 0
                scale0 = 0.1

                som.train(epochs=epochs, radius0=radius0, scale0=scale0)

                '''----------------------clustering params-----------'''
                clusterAlgLabel = 'AffinityPropagation' # KMeans8 , SpectralClustering,AffinityPropagation, Birch 
                if clusterAlgLabel == 'Birch':
                    algorithm = clusterAlgs.Birch()
                elif clusterAlgLabel == 'AffinityPropagation':   
                    original_shape = som.codebook.shape
                    som.codebook.shape = (som._n_columns*som._n_rows, som.n_dim)
                    init = -np.max(distance.pdist(som.codebook, 'euclidean'))      
                    som.codebook.shape = original_shape        
                    algorithm = clusterAlgs.AffinityPropagation(preference = init,damping = 0.9)
                elif clusterAlgLabel == 'KMeans8':
                    algorithm = None

                print('Clustering algorithm employed: %s' %clusterAlgLabel)
                som.cluster(algorithm=algorithm)
                '''----------------------------------------------------'''

                colors = []
                for bm in som.bmus:
                    colors.append(som.clusters[bm[1], bm[0]])
                areas = [200]*len(som.bmus)

                xDimension, yDimension = [], []
                for x in som.bmus:
                    xDimension.append(x[0])
                    yDimension.append(x[1])

                if not os.path.exists(figWritePath+'/'+clusterAlgLabel+'Clusters/'+SOMdimensionsString):
                    os.makedirs(figWritePath+'/'+clusterAlgLabel+'Clusters/'+SOMdimensionsString)
                fig, ax = plt.subplots()
                colMap = 'Spectral_r'
                plt.imshow(som.umatrix,cmap = colMap, aspect = 'auto')
                ax.scatter(xDimension,yDimension,c=colors,s=areas)
                doneLabs = set([''])
                for label, x, y in zip(labels, xDimension, yDimension):
                    lblshiftRatio = 1
                    labFinshift = ''
                    while labFinshift in doneLabs:
                        potentialPositions = [(x, y+lblshiftRatio*lablshift), (x, y-lblshiftRatio*lablshift)]#,(x+lblshiftRatio*lablshift, y+lblshiftRatio*lablshift), 
                        # (x-lblshiftRatio*lablshift, y+lblshiftRatio*lablshift), (x+lblshiftRatio*lablshift, y-lblshiftRatio*lablshift), (x+lblshiftRatio*lablshift, y+lblshiftRatio*lablshift),
                        # (x-lblshiftRatio*lablshift, y+lblshiftRatio*lablshift)]
                        for pP in potentialPositions:
                            labFinshift = pP
                            if labFinshift not in doneLabs:
                                break
                        lblshiftRatio+=1
                    doneLabs.add(labFinshift)
                    plt.annotate(label, xy = (x, y), xytext = labFinshift, textcoords = 'data', ha = 'center', va = 'center', fontsize = 8,bbox = dict(boxstyle = 'round,pad=0.1', fc = 'white', alpha = 0.4))#,arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))

                plt.xlim(0,n_columns)
                plt.ylim(0,n_rows) 
                plt.gca().invert_yaxis()
                plt.xlabel('ESOM')
                mng = plt.get_current_fig_manager()
                mng.window.state('zoomed')
                interactive(True)
                plt.show()            
                fig.savefig(figWritePath+'/'+clusterAlgLabel+'Clusters/'+SOMdimensionsString+'/'+UCRELformat+'SOM_occMat'+years+lvl+'_'+periodIdx+'.png',bbox_inches='tight')
                plt.close()
                interactive(False)
                #-----------------------------------------------------------------------------------------------   
                #-------------------------------------------------------------------------------------------
                '''extract matrix and distance matrix from panda format'''    
                data_array = np.matrix(df)
                data_array_shape = data_array.shape
                data_array_Trans = data_array.T
                data_array_Trans_shape = data_array_Trans.shape
                # data_array = data_array.T
                data_dist = pdist(data_array,'euclidean') # computing the distance
                data_dist_Trans = pdist(data_array_Trans,'euclidean')

                '''Create dendrograms'''            
                if not os.path.exists(figWritePath+'/hierarchicalClustering'):
                    os.makedirs(figWritePath+'/hierarchicalClustering')
                plt.figure()
                plt.xlabel('Term')
                plt.ylabel('Mean')
                plt.title('euclidean - farthest neighbor HCA')
                dendrogram(sch.linkage(data_dist, method='complete'),labels=list(df.index),leaf_rotation=90.,leaf_font_size=leaffont[lIdx],orientation = 'top',truncate_mode = 'none',show_leaf_counts=True)#leaf_font_size=leaffont[lIdx]
                ax = plt.gca()
                ax.set_ylim(-0.01,ax.get_ylim()[1])
                mng = plt.get_current_fig_manager()
                mng.window.state('zoomed')
                interactive(True)
                plt.show()      
                plt.savefig(figWritePath+'/hierarchicalClustering/'+UCRELformat+'DendroCategory'+years+lvl+'_'+periodIdx+'.pdf',bbox_inches='tight')
                plt.close()
                interactive(False)

                # plt.figure()
                # plt.xlabel('Artwork')
                # plt.ylabel('Mean')
                # plt.title('euclidean - farthest neighbor HCA')
                # dendrogram(sch.linkage(data_dist_Trans, method='complete'),labels=list(df.columns),leaf_rotation=90.,orientation = 'top',truncate_mode = 'none',show_leaf_counts=True)#leaf_font_size=leaffont[lIdx]
                # ax = plt.gca()
                # ax.set_ylim(-0.01,ax.get_ylim()[1])
                # mng = plt.get_current_fig_manager()
                # mng.window.state('zoomed')
                # interactive(True)
                # plt.show()      
                # plt.savefig(figWritePath+'/hierarchicalClustering/'+UCRELformat+'DendroArtwork'+years+lvl+'_'+periodIdx+'.pdf',bbox_inches='tight')
                # plt.close()
                # interactive(False)

                '''Create dendroheatmap'''
                # # Compute and plot first dendrogram.
                # fig = plt.figure()#figsize=(8,8)
                # ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
                # side_dendrogram = sch.linkage(data_array_Trans, method='complete')
                # Z1 = sch.dendrogram(side_dendrogram, orientation='left',truncate_mode = 'none',show_leaf_counts=False,no_labels = True)
                # ax1.set_ylim(-0.01,ax1.get_ylim()[1])
                # ax1.set_xticks([])
                # ax1.set_yticks([])

                # # Compute and plot second dendrogram.
                # ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
                # top_dendrogram = sch.linkage(data_dist, method='complete')
                # Z2 = sch.dendrogram(top_dendrogram, truncate_mode = 'none',show_leaf_counts=False,no_labels = True)
                # ax2.set_ylim(-0.01,ax2.get_ylim()[1])
                # ax2.set_xticks([])
                # ax2.set_yticks([])

                # # Plot distance matrix.
                # axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
                # idx1 = Z1['leaves']
                # idx2 = Z2['leaves']
                # data_array = data_array[:,idx1]
                # data_array = data_array[idx2,:]
                # ylabels = df.columns[idx1]
                # xlabels = df.index[idx2]
                # xlabels = [x.replace('_',' ') for x in xlabels]
                # newylabels = []
                # for yl in ylabels:
                #     yl = yl.strip().split(' ')
                #     if len(yl)>5:
                #         yl.insert(3,'\n')
                #     yl = ' '.join(yl)
                #     newylabels.append(yl)
                # axmatrix.matshow(data_array.T, aspect='auto', cmap=plt.cm.Spectral_r, origin='lower', vmin=0, vmax=1)#) 
                # axmatrix.xaxis.tick_bottom()
                # axmatrix.set_xticks(np.arange(len(df.index)))
                # axmatrix.yaxis.tick_right()
                # axmatrix.set_yticks(np.arange(len(df.columns)))
                # axmatrix.set_xticklabels(xlabels, fontsize=10, rotation = 90)
                # axmatrix.set_yticklabels(newylabels,fontsize=10)#minor=False,
                # axmatrix.grid(True,color='gray')

                # mng = plt.get_current_fig_manager()
                # mng.window.state('zoomed')
                # interactive(True)
                # fig.show()
                # fig.savefig(figWritePath+'/hierarchicalClustering/'+UCRELformat+'DendroHeatmap'+years+lvl+'_'+periodIdx+'.pdf',bbox_inches='tight')
                # plt.close()
                # interactive(False)

                #-----------------------------------------------------------------------------------------------
                '''Check for merges and splits'''#-------------------------------------------
                #-----------------------------------------------------------------------------------------------
                if int(periodIdx)>0:
                    if not os.path.exists(splitsMergesWritePath+'/'+SOMdimensionsString):
                        os.makedirs(splitsMergesWritePath+'/'+SOMdimensionsString)
                    tmpStrClusters = [','.join([str(y) for y in x]) for x in som.bmus]
                    strClustDict[periodIdx] = {}
                    for idx, sC in enumerate(tmpStrClusters):
                        if sC in strClustDict[periodIdx]:
                            strClustDict[periodIdx][sC].append(nodes[idx])
                        else:
                            strClustDict[periodIdx][sC] = [nodes[idx]]
                    tmpSameBMUsNodes = list(strClustDict[periodIdx].values())
                    invStrClustDict[periodIdx] = {','.join(v):k for k,v in strClustDict[periodIdx].items()}
                    dataDict[periodIdx]['bmuNodes'] = tmpSameBMUsNodes
                    tmpsplits,tmpmerges = 0, 0
                    with open(splitsMergesWritePath+'/'+SOMdimensionsString+'/'+UCRELformat+'Changes'+years+lvl+'_'+periodIdx+'.txt','w') as f:
                        for tsbn in tmpSameBMUsNodes:
                            if tsbn not in dataDict[str(int(periodIdx)-1)]['bmuNodes']:
                                oldbmucoords = []
                                for ts in tsbn:
                                    for ots in dataDict[str(int(periodIdx)-1)]['bmuNodes']:
                                        if ts in ots:
                                            oldbmucoords.append(invStrClustDict[str(int(periodIdx)-1)][','.join(ots)])
                                if len(set(oldbmucoords)) < 2:
                                    f.write('Terms %s at %s were split from %s \n' %(','.join(tsbn),invStrClustDict[periodIdx][','.join(tsbn)],'|'.join(oldbmucoords)))
                                    if len(tsbn) <= len(strClustDict[str(int(periodIdx)-1)][oldbmucoords[0]])/2:
                                        tmpsplits+=len(tsbn)
                                        termDislocation['splits'].extend(tsbn)
                                        termDislocation['both'].extend(tsbn)
                                else:
                                    f.write('Terms %s at %s were merged from %s \n' %(','.join(tsbn),invStrClustDict[periodIdx][','.join(tsbn)],'|'.join(oldbmucoords)))
                                    for tmpclusts in [strClustDict[str(int(periodIdx)-1)][x] for x in set(oldbmucoords)]:
                                        tmpclustIntersect = set(tmpclusts).intersection(set(tsbn))
                                        if len(tmpclustIntersect) <= len(tsbn)/2:
                                            tmpmerges+=len(tmpclustIntersect)
                                            termDislocation['merges'].extend(tmpclustIntersect)
                                            termDislocation['both'].extend(tmpclustIntersect)
                                # termDislocation['both'].extend(tsbn)
                    dislocationDict['merges'].append(100*tmpmerges/len(dataDict['uniquePersistentUCRELs']))
                    dislocationDict['splits'].append(100*tmpsplits/len(dataDict['uniquePersistentUCRELs']))
                    dislocationDict['both'].append(100*(tmpmerges+tmpsplits)/len(dataDict['uniquePersistentUCRELs']))
                else:
                    tmpStrClusters = [','.join([str(y) for y in x]) for x in som.bmus]
                    strClustDict = {periodIdx:{}}
                    for idx, sC in enumerate(tmpStrClusters):
                        if sC in strClustDict[periodIdx]:
                            strClustDict[periodIdx][sC].append(nodes[idx])
                        else:
                            strClustDict[periodIdx][sC] = [nodes[idx]]
                    dataDict[periodIdx]['bmuNodes'] = list(strClustDict[periodIdx].values())                
                    invStrClustDict = {periodIdx:{','.join(v):k for k,v in strClustDict[periodIdx].items()}}
                    dislocationDict = {'merges':[],'splits':[],'both':[]}
                    termDislocation = {'merges':[],'splits':[],'both':[]}
                #-------------------------------------------------------------------------------------------------------------------------------------
 

elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
