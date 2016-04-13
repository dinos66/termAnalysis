# -*- coding: utf-8 -*-
'''
   Term cluster evolution detection
'''
print('Term cluster evolution detection and term similarity estimation')
#--------------------------------------------
import glob, pickle, pprint, random, re, itertools, time, os
from nltk.corpus import wordnet as wn
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import interactive
import scipy.stats as stats
import numpy as np
from wordcloud import WordCloud
import shutil
#--------------------------------------------

def wordnetClusterSimilarityComputer(clusterTerms):
    allcombs = list(itertools.combinations(clusterTerms,2))
    maxSimilarities = []
    for tt in allcombs:
        ss1 = wn.synsets(tt[0])
        ss2 = wn.synsets(tt[1])
        try:
            maxSimilarities.append(max([x for x in [s1.wup_similarity(s2, simulate_root=False) for (s1, s2) in product(ss1, ss2)] if x]))
        except:
            pass
    if not maxSimilarities:
        maxSimilarities = [0]
    return sum(maxSimilarities)/len(list(allcombs))

t = time.time()

LVLs = ['lvl1','lvl2','lvl3','lvlA']#['lvl1','lvl2','lvl3','lvlA'] #'lvl1','lvl2','lvl3',
yearPeriods = ['1800s','2000s'] 
trueYearsIni = [1800,1964]#
clusterAlgLabel = 'AffinityPropagation'
figWritePath = './data/artworks_figs'


thres = 0.5
averageSizeDict, clusterWordNetSimDict = {}, {}
bmuAverageSizeDict, bmuClusterWordNetSimDict = {}, {}
randomCluster, bmuRandomCluster = {}, {}
for lIdx,lvl in enumerate(LVLs):
    print(lvl)
    averageSizeDict[lvl], clusterWordNetSimDict[lvl] = {}, {}
    bmuAverageSizeDict[lvl], bmuClusterWordNetSimDict[lvl] = {}, {}
    randomCluster[lvl], bmuRandomCluster[lvl] = {}, {}
    for idy,years in enumerate(yearPeriods):
        print(years)
        averageSizeDict[lvl][years], clusterWordNetSimDict[lvl][years] = [], {}
        bmuAverageSizeDict[lvl][years], bmuClusterWordNetSimDict[lvl][years] = [], {}
        randomCluster[lvl][years], bmuRandomCluster[lvl][years] = {}, {}
        n_columns, n_rows = 200, 120
        # n_columns, n_rows = 150, 90
        # n_columns, n_rows = 100, 60
        # if lvl == 'lvl1':
        #  n_columns, n_rows = 20, 12
        # elif lvl == 'lvl2':
        #  n_columns, n_rows = 40, 24
        # elif lvl == 'lvl3':
        #  n_columns, n_rows = 50, 30
        # elif lvl == 'lvlA':
        #  n_columns, n_rows = 60, 40
        SOMdimensionsString = 'x'.join([str(x) for x in [n_columns,n_rows]])
        print('SOM dimension is: %s' %SOMdimensionsString)
        textFilesPath = figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/textFiles/'+SOMdimensionsString+'/'+years+lvl
        if not os.path.exists(textFilesPath):
            os.makedirs(textFilesPath)
        datasetPath = figWritePath+'/'+clusterAlgLabel+'Clusters/bagsOfClusteredTerms/'+SOMdimensionsString+'/boct'+years+lvl+'_'
        bmuDatasetPath = figWritePath+'/bagsOfBMUclusteredTerms/'+SOMdimensionsString+'/boct'+years+lvl+'_'
        files = glob.glob(datasetPath + '*.tsv')
        clusterDict, termDict = {}, {}
        bmuClusterDict, bmuTermDict = {}, {}
        myplot, randplot = [], []
        bmuMyplot, bmuRandplot = [], []
        for timeslot in range(len(files)):
            strTimeSl = str(timeslot)
            clusterDict[timeslot] = {}
            with open(datasetPath+strTimeSl+'.tsv','r') as f:
                for line in f:
                    l = line.strip().split('\t')
                    terms = set(l[1].split(','))
                    clusterDict[timeslot][l[0]] = terms
                    if len(terms) > 1:
                        for te in terms:
                            if te in termDict:
                                termDict[te].append(l[0])
                            else:
                                termDict[te] = [l[0]]
            bmuClusterDict[timeslot] = {}
            with open(bmuDatasetPath+strTimeSl+'.tsv','r') as f:
                for line in f:
                    l = line.strip().split('\t')
                    terms = set(l[1].split(','))
                    bmuClusterDict[timeslot][l[0]] = terms
                    if len(terms) > 1:
                        for te in terms:
                            if te in bmuTermDict:
                                bmuTermDict[te].append(l[0])
                            else:
                                bmuTermDict[te] = [l[0]]
            if timeslot > 0:
                prevTmslKeys = list(clusterDict[timeslot-1].keys())
                currentTmslKeys = list(clusterDict[timeslot].keys())
                allTermsInClusters, allClusterSizes = [], []
                clusterWordNetSimDict[lvl][years][timeslot] = []
                for cK in currentTmslKeys:
                    if len(clusterDict[timeslot][cK]) > 1:
                        averageSizeDict[lvl][years].append(len(clusterDict[timeslot][cK]))
                        allClusterSizes.append(len(clusterDict[timeslot][cK]))
                        # clusterWordNetSimDict[lvl][years][timeslot].append(wordnetClusterSimilarityComputer([" ".join(re.findall("[a-zA-Z]+", x)) for x in clusterDict[timeslot][cK]]))
                        allTermsInClusters.extend(clusterDict[timeslot][cK])
                        for pK in prevTmslKeys:
                            if len(clusterDict[timeslot-1][pK]) > 1:
                                intersLen = len(clusterDict[timeslot][cK].intersection(clusterDict[timeslot-1][pK]))
                                sim = intersLen / len(clusterDict[timeslot][cK].union(clusterDict[timeslot-1][pK]))
                                if sim > thres:
                                    tmpprevclusterLabel = ','.join([str(x) for x in [timeslot-1,pK]])
                                    tmpclusterLabel = ','.join([str(x) for x in [timeslot,cK]])
                                    if tmpprevclusterLabel in invEvolContainer:
                                        wordcloud = WordCloud().generate(' '.join(sorted(list(clusterDict[timeslot][cK]))))
                                        for invCounter in invEvolContainer[tmpprevclusterLabel]:
                                            evolContainer[invCounter].append(tmpclusterLabel)
                                            if tmpclusterLabel in invEvolContainer:
                                                invEvolContainer[tmpclusterLabel].append(invCounter)
                                            else:
                                                invEvolContainer[tmpclusterLabel] = [invCounter]
##                                            wordcloud = WordCloud().generate(' '.join(sorted(list(clusterDict[timeslot][cK]))))

                                            with open(textFilesPath+'/'+str(invCounter)+'.txt','a') as tf:
                                                tf.write(tmpclusterLabel+'\t'+','.join(sorted(list(clusterDict[timeslot][cK])))+'\n')

                                            if not os.path.exists(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(invCounter)):
                                                os.makedirs(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(invCounter))
                                            interactive(True)
                                            plt.imshow(wordcloud)
                                            plt.axis("off")
                                            plt.savefig(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(invCounter)+'/'+tmpclusterLabel+'.png',bbox_inches='tight')
                                            plt.close()
                                            interactive(False)                                        
                                    else:
                                        counter = len(evolContainer)
                                        evolContainer[counter] = [tmpprevclusterLabel,tmpclusterLabel]
                                        invEvolContainer[tmpprevclusterLabel] = [counter]
                                        if tmpclusterLabel in invEvolContainer:
                                            invEvolContainer[tmpclusterLabel].append(counter)
                                        else:
                                            invEvolContainer[tmpclusterLabel] = [counter]

                                        with open(textFilesPath+'/'+str(counter)+'.txt','w') as tf:
                                            tf.write(tmpprevclusterLabel+'\t'+','.join(sorted(list(clusterDict[timeslot-1][pK])))+'\n')
                                            tf.write(tmpclusterLabel+'\t'+','.join(sorted(list(clusterDict[timeslot][cK])))+'\n')

                                        wordcloud1 = WordCloud().generate(' '.join(sorted(list(clusterDict[timeslot-1][pK]))))
                                        wordcloud2 = WordCloud().generate(' '.join(sorted(list(clusterDict[timeslot][cK]))))
                                        if not os.path.exists(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(counter)):
                                            os.makedirs(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(counter))
                                        interactive(True)
                                        plt.imshow(wordcloud1)
                                        plt.axis("off")
                                        plt.savefig(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(counter)+'/'+tmpprevclusterLabel+'.png',bbox_inches='tight')
                                        plt.close()
                                        interactive(False)
                                        interactive(True)
                                        plt.imshow(wordcloud2)
                                        plt.axis("off")
                                        plt.savefig(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(counter)+'/'+tmpclusterLabel+'.png',bbox_inches='tight')
                                        plt.close()
                                        interactive(False)
                                        
                allTermsInClusters = list(set(allTermsInClusters))
                randomCluster[lvl][years][timeslot] = []
                for cS in allClusterSizes:
                    randomTermIndices = random.sample(range(len(allTermsInClusters)),cS)
                    tmpRandTerms = [allTermsInClusters[i] for i in randomTermIndices]
                    # randomCluster[lvl][years][timeslot].append(wordnetClusterSimilarityComputer([" ".join(re.findall("[a-zA-Z]+", x)) for x in tmpRandTerms]))
                    allTermsInClusters = [x for x in allTermsInClusters if x not in tmpRandTerms]
                '''bmu cluster similarity analysis'''
                bmuCurrentTmslKeys = list(bmuClusterDict[timeslot].keys())
                allTermsInClusters, allClusterSizes = [], []
                bmuClusterWordNetSimDict[lvl][years][timeslot] = []
                for cK in bmuCurrentTmslKeys:
                    if len(bmuClusterDict[timeslot][cK]) > 1:
                        bmuAverageSizeDict[lvl][years].append(len(bmuClusterDict[timeslot][cK]))
                        allTermsInClusters.extend(bmuClusterDict[timeslot][cK])
                        allClusterSizes.append(len(bmuClusterDict[timeslot][cK]))
                        # bmuClusterWordNetSimDict[lvl][years][timeslot].append(wordnetClusterSimilarityComputer([" ".join(re.findall("[a-zA-Z]+", x)) for x in bmuClusterDict[timeslot][cK]]))
                allTermsInClusters = list(set(allTermsInClusters))
                bmuRandomCluster[lvl][years][timeslot] = []
                for cS in allClusterSizes:
                    randomTermIndices = random.sample(range(len(allTermsInClusters)),cS)
                    tmpRandTerms = [allTermsInClusters[i] for i in randomTermIndices]
                    # bmuRandomCluster[lvl][years][timeslot].append(wordnetClusterSimilarityComputer([" ".join(re.findall("[a-zA-Z]+", x)) for x in tmpRandTerms]))
                    allTermsInClusters = [x for x in allTermsInClusters if x not in tmpRandTerms]
            else:
                evolContainer = {}
                invEvolContainer = {}
                currentTmslKeys = list(clusterDict[timeslot].keys())
                allTermsInClusters, allClusterSizes = [], []
                clusterWordNetSimDict[lvl][years][timeslot] = []
                for cK in currentTmslKeys:
                    if len(clusterDict[timeslot][cK]) > 1:
                        counter = len(evolContainer)
                        tmpclusterLabel = ','.join([str(x) for x in [timeslot,cK]])
                        evolContainer[counter] = [tmpclusterLabel]
                        invEvolContainer[tmpclusterLabel] = [counter]
                        averageSizeDict[lvl][years].append(len(clusterDict[timeslot][cK]))
                        allClusterSizes.append(len(clusterDict[timeslot][cK]))
                        # clusterWordNetSimDict[lvl][years][timeslot].append(wordnetClusterSimilarityComputer([" ".join(re.findall("[a-zA-Z]+", x)) for x in clusterDict[timeslot][cK]]))
                        allTermsInClusters.extend(clusterDict[timeslot][cK])
                        if not os.path.exists(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(counter)):
                            os.makedirs(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(counter))
                        with open(textFilesPath+'/'+str(counter)+'.txt','w') as tf:
                            tf.write(tmpclusterLabel+'\t'+','.join(sorted(list(clusterDict[timeslot][cK])))+'\n')
                        wordcloud = WordCloud().generate(' '.join(sorted(list(clusterDict[timeslot][cK]))))
                        interactive(True)
                        plt.imshow(wordcloud)
                        plt.axis("off")
                        plt.savefig(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(counter)+'/'+tmpclusterLabel+'.png',bbox_inches='tight')
                        plt.close()
                        interactive(False)
                allTermsInClusters = list(set(allTermsInClusters))
                randomCluster[lvl][years][timeslot] = []
                for cS in allClusterSizes:
                    randomTermIndices = random.sample(range(len(allTermsInClusters)),cS)
                    tmpRandTerms = [allTermsInClusters[i] for i in randomTermIndices]
                    # randomCluster[lvl][years][timeslot].append(wordnetClusterSimilarityComputer([" ".join(re.findall("[a-zA-Z]+", x)) for x in tmpRandTerms]))
                    allTermsInClusters = [x for x in allTermsInClusters if x not in tmpRandTerms]
                '''bmu cluster similarity analysis'''
                bmuCurrentTmslKeys = list(bmuClusterDict[timeslot].keys())
                allTermsInClusters, allClusterSizes = [], []
                bmuClusterWordNetSimDict[lvl][years][timeslot] = []
                for cK in bmuCurrentTmslKeys:
                    if len(bmuClusterDict[timeslot][cK]) > 1:
                        bmuAverageSizeDict[lvl][years].append(len(bmuClusterDict[timeslot][cK]))
                        allClusterSizes.append(len(bmuClusterDict[timeslot][cK]))
                        # bmuClusterWordNetSimDict[lvl][years][timeslot].append(wordnetClusterSimilarityComputer([" ".join(re.findall("[a-zA-Z]+", x)) for x in bmuClusterDict[timeslot][cK]]))
                        allTermsInClusters.extend(bmuClusterDict[timeslot][cK])
                allTermsInClusters = list(set(allTermsInClusters))
                bmuRandomCluster[lvl][years][timeslot] = []
                for cS in allClusterSizes:
                    randomTermIndices = random.sample(range(len(allTermsInClusters)),cS)
                    tmpRandTerms = [allTermsInClusters[i] for i in randomTermIndices]
                    # bmuRandomCluster[lvl][years][timeslot].append(wordnetClusterSimilarityComputer([" ".join(re.findall("[a-zA-Z]+", x)) for x in tmpRandTerms]))
                    allTermsInClusters = [x for x in allTermsInClusters if x not in tmpRandTerms]
            '''plot data'''
            try:
                myplot.append(sum(clusterWordNetSimDict[lvl][years][timeslot])/len(clusterWordNetSimDict[lvl][years][timeslot]))
                randplot.append(sum(randomCluster[lvl][years][timeslot])/len(randomCluster[lvl][years][timeslot]))
                randh = sorted(clusterWordNetSimDict[lvl][years][timeslot])
                fit = stats.norm.pdf(randh, np.mean(randh), np.std(randh))
                # plt.plot(randh,fit,'-o')
                # plt.hist(randh,normed=True) 
                # plt.show()

            except:
                myplot.append(0)
                randplot.append(0)
                pass
            try:
                bmuMyplot.append(sum(bmuClusterWordNetSimDict[lvl][years][timeslot])/len(bmuClusterWordNetSimDict[lvl][years][timeslot]))
                bmuRandplot.append(sum(bmuRandomCluster[lvl][years][timeslot])/len(bmuRandomCluster[lvl][years][timeslot]))
            except:
                bmuMyplot.append(0)
                bmuRandplot.append(0)
                pass
        '''make figures'''
        if not os.path.exists(figWritePath+'/'+clusterAlgLabel+'Clusters/semanticSimilarity/'+SOMdimensionsString):
            os.makedirs(figWritePath+'/'+clusterAlgLabel+'Clusters/semanticSimilarity/'+SOMdimensionsString)
        # plt.figure()
        # plt.xlabel('Timeslot')
        # plt.ylabel('Average Path Similarity (WordNet)')
        # plt.title('Term average path similarity (WordNet) per timeslot (Level '+lvl+' terms | 5 year period prior to '+str(timeslot*5+trueYearsIni[idy])+')')
        # plt.plot(myplot,marker='*',label = 'Somoclu - AffinityPropagation')
        # plt.plot(randplot,marker='*',label = 'random')
        # plt.plot(bmuMyplot,marker='o',linestyle = '--',label = 'Somoclu - BMU')
        # plt.plot(bmuRandplot,marker='o',linestyle = '--',label = 'random - BMU')
        # plt.legend()
        # mng = plt.get_current_fig_manager()
        # mng.window.state('zoomed')
        # interactive(True)
        # plt.show()            
        # plt.savefig(figWritePath+'/'+clusterAlgLabel+'Clusters/semanticSimilarity/'+SOMdimensionsString+'/semSim'+years+lvl+'.png',bbox_inches='tight')
        # plt.close()
        # interactive(False)

        averageSizeDict[lvl][years] = sum(averageSizeDict[lvl][years])/len(averageSizeDict[lvl][years])
        bmuAverageSizeDict[lvl][years] = sum(bmuAverageSizeDict[lvl][years])/len(bmuAverageSizeDict[lvl][years])

        for c in range(counter):
            if len(evolContainer[c]) < 2:
                del(evolContainer[c])
                shutil.rmtree(figWritePath+'/'+clusterAlgLabel+'Clusters/evolutionAnalysis/wordcloud/'+SOMdimensionsString+'/'+years+lvl+'/'+str(c))
                os.remove(textFilesPath+'/'+str(c)+'.txt')

# pprint.pprint(termDict)
print('-------------------------------------------------------------------------')
# pprint.pprint(evolContainer)
print('-------------------------------------------------------------------------')
# pprint.pprint(invEvolContainer)
print('-------------------------------------------------------------------------')
pprint.pprint(averageSizeDict)

elapsed = time.time() - t
print('Total time Elapsed: %.2f seconds' % elapsed)
