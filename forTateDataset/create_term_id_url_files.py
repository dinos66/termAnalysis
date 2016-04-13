# -*- coding: utf-8 -*-
'''
    Create tsvs with term id url files
'''
print('Create tsvs with term id url files')
#---------------------------------------
#this py is followed by termAnalysis.py
#---------------------------------------
import os, codecs, pickle
#---------------------------------------

globalTermDict = pickle.load(open('./data/artworks_tmp/globalTermDict.pck','rb'))

statsWritePath = './data/artworks_stats'
targetpath = './data/artworks_verification_labels/Term_Id_URL files'
if not os.path.exists(targetpath):
    os.makedirs(targetpath)

LVLs = ['lvl1','lvl2','lvl3'] #'lvl1','lvl2','lvl3',
yearPeriods = ['1800s','2000s'] 

for idy,years in enumerate(yearPeriods):    
    for lvl in LVLs:
        persistTerms = [x.strip() for x in open(statsWritePath+'/'+years+lvl+'_unique_persistent_terms.txt','r').readlines()]
        with open(targetpath+'/persistentTermIdsURLs'+years+lvl+'.tsv','w') as f:
            f.write('Term\tURLs\n')
            for t in persistTerms:                
                f.write(t+'\t'+', '.join(globalTermDict[t]['URLs'][:10])+'\n')




