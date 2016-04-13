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
'''Select ucrel format to use: label or code'''
UCRELformat = 'label' # code 

targetpath = './data/artworks_tmp/artworks_doc_UCREL'
tsvReadPath = './other_data/artworks_TSVs/dynamic'
statsWritePath = './data/artworks_stats'
if not os.path.exists(targetpath):
    os.makedirs(targetpath)

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
        persistUCREL = list(set([termLabelDict[" ".join(re.findall("[a-zA-Z]+", x))][UCRELformat] for x in persistTerms]))
        persistUCREL.sort()
        with open(statsWritePath+'/'+years+lvl+'_unique_persistent_UCREL'+UCRELformat+'.txt','w') as f:
            for word in persistUCREL:
                f.write(word+'\n') 
        for periodIdx in yearList:
            with open(targetpath+'/'+UCRELformat+'OccurrenceMat'+years+lvl+'_'+periodIdx+'.tsv','w') as f:
                f.write('Term\t'+'\t'.join(persistUCREL)+'\n')
                for o in objects[periodIdx]:
                    tmp = [0]*len(persistUCREL)
                    for t in dataDict[periodIdx][o][lvl]:
                        if t in persistTerms:
                            tmp[persistUCREL.index(termLabelDict[" ".join(re.findall("[a-zA-Z]+", t))][UCRELformat])] += 1
                    f.write(o+'\t'+'\t'.join([str(x) for x in tmp])+'\n')

