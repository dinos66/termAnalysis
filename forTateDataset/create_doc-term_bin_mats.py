# -*- coding: utf-8 -*-
'''
    Create tsv, edge files and adjacency matrices
'''
print('Create tsv and edge files')
#---------------------------------------
#this py is followed by termAnalysis.py
#---------------------------------------
import json, sys, pprint, time, os, codecs, glob, pickle, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import interactive

#---------------------------------------

targetpath = './data/artworks_tmp/artworks_doc_term'
tsvReadPath = './other_data/artworks_TSVs/dynamic'
statsWritePath = './data/artworks_stats'
figWritePath = './data/artworks_figs/hierarchicalClustering'
if not os.path.exists(targetpath):
    os.makedirs(targetpath)
if not os.path.exists(figWritePath):
    os.makedirs(figWritePath)

LVLs = ['lvl1','lvl2','lvl3','lvlA'] #'lvl1','lvl2','lvl3',
yearPeriods = ['1800s','2000s'] 
trueYearsIni = [1800,1964]
leaffont = [12,8,6,4]#12,7,6,

termLabelDict = pickle.load(open('./data/artworks_verification_labels/WmatrixLabelDict.pck','rb'))

for idy,years in enumerate(yearPeriods):
    print(years)
    files = glob.glob(tsvReadPath+'/dyn'+years+'_*.tsv')
    files.sort(key=lambda x: os.path.getmtime(x))
    yearList = []
    dataDict = {}
    objects = {}
    for filename in files:
        periodIdx = filename[filename.index('0s_')+3:-4]  
        dataDict[periodIdx] = {}      
        yearList.append(periodIdx)
        allLines = [x.strip() for x in codecs.open(filename,'r','utf8').readlines()]
        objects[periodIdx] = []
        for l in allLines:
            line = l.split('\t')
            lvl1, lvl2, lvl3 = line[-3].split(','), line[-2].split(','), line[-1].split(',')
            lvlA = ['1-'+x for x in lvl1] + ['2-'+x for x in lvl2] + ['3-'+x for x in lvl3]
            dataDict[periodIdx][line[0]] = {'lvl1':lvl1,'lvl2':lvl2,'lvl3':lvl3,'lvlA':lvlA}
            objects[periodIdx].append(line[0])
    
    
    for lIdx,lvl in enumerate(LVLs):
        print(lvl)
        persistTerms = [x.strip() for x in open(statsWritePath+'/'+years+lvl+'_unique_persistent_terms.txt','r').readlines()]
        for periodIdx in yearList:
            data_array = []
            with open(targetpath+'/binMat'+years+lvl+'_'+periodIdx+'.tsv','w') as f:
                f.write('Term\t'+'\t'.join(persistTerms)+'\n')
                for o in objects[periodIdx]:
                    tmp = []
                    for t in persistTerms:
                        if t in dataDict[periodIdx][o][lvl]:
                            tmp.append('1')
                        else:
                            tmp.append('0')
                    f.write(o+'\t'+'\t'.join(tmp)+'\n')
                    data_array.append([int(x) for x in tmp])

            data_array = np.matrix(data_array)
            data_array = data_array.T
            data_dist = pdist(data_array,'euclidean') # computing the distance

            labels = [x+'_'+termLabelDict[" ".join(re.findall("[a-zA-Z]+", x))]['code'] for x in persistTerms]
            plt.figure()
            plt.xlabel('Terms')
            plt.ylabel('Distance')
            plt.title('euclidean - farthest neighbor HCA  (Level '+lvl+' terms | 5 year period prior to '+str(int(periodIdx)*5+trueYearsIni[idy])+')')
            dendrogram(linkage(data_dist, method='complete'),labels=labels,leaf_rotation=90.,leaf_font_size=leaffont[lIdx],orientation = 'top',truncate_mode = 'none',show_leaf_counts=True)#
            ax = plt.gca()
            ax.set_ylim(-1,ax.get_ylim()[1])
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            interactive(True)
            plt.show()      
            plt.savefig(figWritePath+'/dendro'+years+lvl+'_'+periodIdx+'.pdf',bbox_inches='tight')
            plt.close()
            interactive(False)




