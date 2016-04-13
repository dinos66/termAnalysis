'''
    Create tsv, edge files and adjacency matrices
'''
print('Create tsv and edge files')
#---------------------------------------
#this py is followed by termAnalysis.py
#---------------------------------------
import json, sys, pprint, time, os, codecs, collections, itertools, pickle
from os import walk
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
from itertools import combinations
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#---------------------------------------
#parameters-----------------------------
samplingTimeinYears = 5
# --------------------------------------

targetpath = './artworks'
tsvWritePath = './other_data/artworks_TSVs'
edgeWritePath = './data/artworks_tmp/edges'
statsWritePath = './data/artworks_stats'
tmpWritePath = './data/artworks_tmp'
if not os.path.exists(tsvWritePath+'/dynamic'):
    os.makedirs(tsvWritePath+'/dynamic')
if not os.path.exists(edgeWritePath+'/dynamic'):
    os.makedirs(edgeWritePath+'/dynamic')
if not os.path.exists(statsWritePath):
    os.makedirs(statsWritePath)
if not os.path.exists(tmpWritePath):
    os.makedirs(tmpWritePath)


dataDict, fullDataDict = {}, {}
idList, fullIdList = [], []
dateList, fullDateList = [], []

try:
    stopW = stopwords.words('english')
except:
    nltk.download('stopwords')
    stopW = stopwords.words('english')
    pass
# stopW.extend(stopwords.words('french'))
# stopW.extend(stopwords.words('italian'))
# stopW.extend(stopwords.words('spanish'))
# stopW.extend(stopwords.words('german'))
# stopW.extend(stopwords.words('dutch'))    
# stopW.extend(stopwords.words('german'))
# stopW.extend(stopwords.words('portuguese'))
stopW.extend(['non','ii','iii','iv','viii','vii','xii','xiii','xv'])
# stopW.extend(['le','la','de','ss','au'])
stopW = set(stopW)
# tokenizer = RegexpTokenizer(r'\w+')
tokenizer = RegexpTokenizer(r"\b[^\d\W]+\b")

allcount, okcount,classificationCount = 0, 0, 0
globalTermDict = {}
rawLvl1,rawLvl2,rawLvl3=[],[],[]
for dirname, dirnames, filenames in walk(targetpath):

    for filename in filenames:
        filepath = '/'.join([dirname,filename])
        fileopen = open(filepath).read()
        jsonfile = json.loads(fileopen)
        allcount+=1

        #-------------------------------------------------------------------
        Id = filename[:-5]#jsonfile['id']
        try:
            date = int(jsonfile['dateRange']['endYear'])
        except:
            if jsonfile['dateRange']:
                try:
                    date = int(jsonfile['dateRange']['startYear'])
                except:
                    continue
                    pass
            else:
                continue
            pass
        fullDataDict[Id] = {'date':date}
        fullIdList.append(Id)
        fullDateList.append(date)
        if jsonfile['classification']:
            fullDataDict[Id]['classification'] = jsonfile['classification']
        else:
            fullDataDict[Id]['classification'] = 'unclassified'
        if jsonfile['title']:
            fullDataDict[Id]['title'] = jsonfile['title']
        else:
            fullDataDict[Id]['title'] = 'untitled'
        #-------------------------------------------------------------------

        subjectcount = jsonfile['subjectCount']

        if subjectcount is not 0:
            Id = filename[:-5]#jsonfile['id']
            try:
                date = int(jsonfile['dateRange']['endYear'])
            except:
                if jsonfile['dateRange']:
                    try:
                        date = int(jsonfile['dateRange']['startYear'])
                    except:
                        continue
                        pass
                else:
                    continue
                pass

            try:
                subjects0 = jsonfile['subjects']['children']
            except:
##                    pprint.pprint(jsonfile['subjects'])
##                    print(jsonfile['title'])
##                    print(jsonfile['url'])
##                    xa=input('continue from bad subject?')
##                    print('++++++++++++++++++++++++++++++++++++++')
                continue
                pass

            dataDict[Id] = {'date':date}
            idList.append(Id)
            dateList.append(date)

            if jsonfile['title']:
                dataDict[Id]['title'] = jsonfile['title']
            else:
                dataDict[Id]['title'] = 'untitled'

            if jsonfile['classification']:
                dataDict[Id]['classification'] = jsonfile['classification']
            else:
                dataDict[Id]['classification'] = 'unclassified'

            subjects1 = []
            dataDict[Id]['lvl1']=[]
            for child in subjects0:
                subjects1.extend(child['children'])
                tmp = [w for w in tokenizer.tokenize(child['name'].lower().replace('non-specific','').replace('non specific','')) if len(w)>1 and not w in stopW]
                rawLvl1.append(child['name'].lower())
                dataDict[Id]['lvl1'].extend(tmp)
                dataDict[Id]['lvl1'] = list(set(dataDict[Id]['lvl1']))
                dataDict[Id]['lvl1'].sort()
            for l1 in dataDict[Id]['lvl1']:
              if l1 in globalTermDict:
                globalTermDict[l1]['workIDs'].append(Id)
                globalTermDict[l1]['URLs'].append(jsonfile['url'])
              else:
                globalTermDict[l1] = {'workIDs':[Id],'URLs':[jsonfile['url']]}

            subjects2 = []
            dataDict[Id]['lvl2']=[]
            for child in subjects1:
                subjects2.extend(child['children'])                
                tmp = [w for w in tokenizer.tokenize(child['name'].lower().replace('non-specific','').replace('non specific','')) if len(w)>1 and not w in stopW]
                rawLvl2.append(child['name'].lower())
                dataDict[Id]['lvl2'].extend(tmp)
                dataDict[Id]['lvl2'] = list(set(dataDict[Id]['lvl2']))
                dataDict[Id]['lvl2'].sort()
            for l2 in dataDict[Id]['lvl2']:
              if l2 in globalTermDict:
                globalTermDict[l2]['workIDs'].append(Id)
                globalTermDict[l2]['URLs'].append(jsonfile['url'])
              else:
                globalTermDict[l2] = {'workIDs':[Id],'URLs':[jsonfile['url']]}

            dataDict[Id]['lvl3']=[]
            for child in subjects2:
                tmp = [w for w in tokenizer.tokenize(child['name'].lower().replace('non-specific','').replace('non specific','')) if len(w)>1 and not w in stopW]
                rawLvl3.append(child['name'].lower())
                dataDict[Id]['lvl3'].extend(tmp) 
                dataDict[Id]['lvl3'] = list(set(dataDict[Id]['lvl3']))
                dataDict[Id]['lvl3'].sort()
            for l3 in dataDict[Id]['lvl3']:
              if l3 in globalTermDict:
                globalTermDict[l3]['workIDs'].append(Id)
                globalTermDict[l3]['URLs'].append(jsonfile['url'])
              else:
                globalTermDict[l3] = {'workIDs':[Id],'URLs':[jsonfile['url']]}

            okcount+=1

fullstatement = 'All docs are %s and all timed docs are %s' %(allcount,okcount)
print(fullstatement)

# pickle.dump(globalTermDict,open('./data/artworks_tmp/globalTermDict.pck','wb'), protocol = 2)

rawLvl1 = '\n'.join(sorted(list(set(rawLvl1)), key = lambda k: len(k), reverse = True))
rawLvl2 = '\n'.join(sorted(list(set(rawLvl2)), key = lambda k: len(k), reverse = True))
rawLvl3 = '\n'.join(sorted(list(set(rawLvl3)), key = lambda k: len(k), reverse = True))
with codecs.open(tmpWritePath+'/rawLvl1.txt','w','utf8') as f:
  f.write(rawLvl1)
with codecs.open(tmpWritePath+'/rawLvl2.txt','w','utf8') as f:
  f.write(rawLvl2)
with codecs.open(tmpWritePath+'/rawLvl3.txt','w','utf8') as f:
  f.write(rawLvl3)

#-----------------------------------------------------------------------#
#--------------------------end of making dataDict-----------------------#
#-----------------------------------------------------------------------#

# zippedall=zip(dateList,idList)
# zippedall=sorted(zippedall)
# dateList, idList = zip(*zippedall)
# dateList, idList = list(dateList), list(idList)
# print('Length of dateList is %s' %len(dateList))

# plt.hist(dateList,bins=len(set(dateList)))
# plt.title("Tate Dataset Date Distribution")
# plt.xlabel("Date")
# plt.ylabel("Frequency")
# plt.xlim(1545, 2011)
# fig = plt.gcf()
# # plot_url = py.plot_mpl(fig, filename='Tate_DB_Date_DistrTokenized')
# # plt.draw()
# # fig.savefig(statsWritePath+'/'+'Tate_DB_Distr_Full.pdf', bbox_inches='tight', format='pdf')
# plt.close()
# plt.close(fig)

# #----------------------------------------
# '''write up a TSV for all the artworks'''
# #----------------------------------------
# zippedall=zip(fullDateList,fullIdList)
# zippedall=sorted(zippedall)
# fullDateList, fullIdList = zip(*zippedall)
# fullDateList, fullIdList = list(fullDateList), list(fullIdList)
# print('Length of fulldateList is %s' %len(fullDateList))
# # try:
# summaryCaptionDict = pickle.load(open('./data/artworks_tmp/summaryCaptionDict.pck','rb'))
# with codecs.open(tsvWritePath+'/fullDatasetContainer.tsv','w','utf8') as f:
#     f.write('Id\tTitle\tDescription\tDate\tCategory\n')
#     for idx,t in enumerate(fullDateList):
#         tmpId = fullIdList[idx]
#         try:
#             tmpSummary = summaryCaptionDict[tmpId]['summary']
#         except:
#             tmpSummary = 'noSummary'
#             pass
#         tmpTSV = '\t'.join([tmpId,fullDataDict[tmpId]['title'],tmpSummary,str(t),fullDataDict[tmpId]['classification']])
#         f.write(tmpTSV+'\n')
# # except:
# #     pass
# #----------------------------------------


# timeIni = 1796
# timedIdSeries1800 = [[]]
# allReducedDates = []
# for idx,t in enumerate(dateList):
#     if t > 1795 and t < 1846:
#         allReducedDates.append(t)
#         timeDif = t-timeIni
#         if timeDif > samplingTimeinYears-1:
#             timeIni = t
#             timedIdSeries1800.append([idList[idx]])
#         else:
#             timedIdSeries1800[-1].append(idList[idx])

# statement = '\nAll 1800s reduced docs are %s' %len(allReducedDates)
# print(statement)
# fullstatement += statement

# plt.hist(allReducedDates,bins=int(len(set(allReducedDates))/samplingTimeinYears))
# plt.title("Tate Dataset Date Distribution 1800s")
# plt.xlabel("Date")
# plt.ylabel("Frequency")
# plt.xlim(1796, 1845)
# fig = plt.gcf()
# # plot_url = py.plot_mpl(fig, filename='Tate_DB_Distr_1800sFinalTokenized')
# # plt.draw()
# # fig.savefig(statsWritePath+'/'+'Tate_DB_Distr_1800sFinal.pdf', bbox_inches='tight', format='pdf')
# plt.close()
# plt.close(fig)

# timeIni = 1960
# timedIdSeries2000 = [[]]
# allReducedDates = []
# for idx,t in enumerate(dateList):
#     if t > 1959 and t < 2010:
#         allReducedDates.append(t)
#         timeDif = t-timeIni
#         if timeDif > samplingTimeinYears-1:
#             timeIni = t
#             timedIdSeries2000.append([idList[idx]])
#         else:
#             timedIdSeries2000[-1].append(idList[idx])

# statement = '\nAll 2000s reduced docs are %s' %len(allReducedDates)
# print(statement)
# fullstatement += statement

# plt.hist(allReducedDates,bins=int(len(set(allReducedDates))/samplingTimeinYears))
# plt.title("Tate Dataset Date Distribution 2000s")
# plt.xlabel("Date")
# plt.ylabel("Frequency")
# plt.xlim(1960, 2009)
# fig = plt.gcf()
# # plot_url = py.plot_mpl(fig, filename='Tate_DB_Distr_2000sFinalTokenized')
# # plt.draw()
# # fig.savefig(statsWritePath+'/'+'Tate_DB_Distr_2000sFinal.pdf', bbox_inches='tight', format='pdf')

# allTimedIdSeries = [timedIdSeries1800,timedIdSeries2000]
# yearPeriods = ['1800s','2000s']

# for yIdx,tmptimedIdSeries in enumerate(allTimedIdSeries):
#     alllvl1,alllvl2,alllvl3,alllvlAll = [], [], [], []
#     Years = yearPeriods[yIdx]
#     timedIdStaticTSV,timedIdStaticEdges1,timedIdStaticEdges2,timedIdStaticEdges3,timedIdStaticEdgesAll = [], [], [], [], []
#     for idx,timePeriod in enumerate(tmptimedIdSeries):
#         with codecs.open(tsvWritePath+'/dynamic/dyn'+Years+'_'+str(idx)+'.tsv','w','utf8') as f:
#             with codecs.open(edgeWritePath+'/dynamic/'+Years+'lvl3_'+str(idx)+'.csv','w','utf8') as f3:
#                 with codecs.open(edgeWritePath+'/dynamic/'+Years+'lvl2_'+str(idx)+'.csv','w','utf8') as f2:
#                     with codecs.open(edgeWritePath+'/dynamic/'+Years+'lvl1_'+str(idx)+'.csv','w','utf8') as f1:
#                       with codecs.open(edgeWritePath+'/dynamic/'+Years+'lvlA_'+str(idx)+'.csv','w','utf8') as fAll:
#                           f1.write('Source,Target,Weight,Type' + '\n')
#                           f2.write('Source,Target,Weight,Type' + '\n')
#                           f3.write('Source,Target,Weight,Type' + '\n')
#                           fAll.write('Source,Target,Weight,Type' + '\n')
#                           tmpTSV, tmpEdges1, tmpEdges2, tmpEdges3, tmpEdgesAll = [], [], [], [], []
#                           for Id in timePeriod:
#                               try:
#                                   lvl1 = ','.join(dataDict[Id]['lvl1'])
#                                   lvl2 = ','.join(dataDict[Id]['lvl2'])
#                                   lvl3 = ','.join(dataDict[Id]['lvl3'])
#                                   tmpTSV.append('\t'.join([str(Id),dataDict[Id]['title'],str(dataDict[Id]['date']),lvl1,lvl2,lvl3]))
#                                   alllvl1.extend(dataDict[Id]['lvl1'])
#                                   alllvl2.extend(dataDict[Id]['lvl2'])                
#                                   alllvl3.extend(dataDict[Id]['lvl3'])
#                                   tmpEdges1.extend([','.join(x) for x in list(combinations(dataDict[Id]['lvl1'],2)) if x[0] != x[1]])
#                                   tmpEdges2.extend([','.join(x) for x in list(combinations(dataDict[Id]['lvl2'],2)) if x[0] != x[1]])
#                                   tmpEdges3.extend([','.join(x) for x in list(combinations(dataDict[Id]['lvl3'],2)) if x[0] != x[1]])
#                                   tmpEdgesAll.extend([','.join(x) for x in list(combinations(list(itertools.chain.from_iterable([['1-'+x for x in dataDict[Id]['lvl1']],['2-'+x for x in dataDict[Id]['lvl2']],['3-'+x for x in dataDict[Id]['lvl3']]])),2)) if x[0] != x[1]])
#                               except Exception as e:
#                                   print(e)
#                                   print(str(Id))
#                                   print(str(dataDict[Id]['date']))
#                                   print(dataDict[Id]['lvl1'])
#                                   print(dataDict[Id]['lvl2'])
#                                   print(dataDict[Id]['lvl3'])
#                                   time.sleep(3)
#                                   print('\n')
#                                   pass
#                           f.writelines('\n'.join(tmpTSV))
#                           timedIdStaticTSV.extend(tmpTSV)
#                           counttmpEdges1 = collections.Counter(tmpEdges1)
#                           truple1 = [','.join([str(y) for y in x])+',undirected' for x in counttmpEdges1.most_common()]
#                           f1.writelines('\n'.join(truple1))
#                           timedIdStaticEdges1.extend(tmpEdges1)
#                           counttmpEdges2 = collections.Counter(tmpEdges2)
#                           truple2 = [','.join([str(y) for y in x])+',undirected' for x in counttmpEdges2.most_common()]
#                           f2.writelines('\n'.join(truple2))
#                           timedIdStaticEdges2.extend(tmpEdges2)
#                           counttmpEdges3 = collections.Counter(tmpEdges3)
#                           truple3 = [','.join([str(y) for y in x])+',undirected' for x in counttmpEdges3.most_common()]
#                           f3.writelines('\n'.join(truple3))
#                           timedIdStaticEdges3.extend(tmpEdges3)
#                           counttmpEdgesAll = collections.Counter(tmpEdgesAll)
#                           trupleAll = [','.join([str(y) for y in x])+',undirected' for x in counttmpEdgesAll.most_common()]
#                           fAll.writelines('\n'.join(trupleAll))
#                           timedIdStaticEdgesAll.extend(tmpEdgesAll)
        


#     with codecs.open(tsvWritePath+'/static'+Years+'.tsv','w','utf8') as f:
#         f.writelines('\n'.join(timedIdStaticTSV))

#     with codecs.open(edgeWritePath+'/staticEdges'+Years+'lvl1.csv','w','utf8') as f:
#         f.write('Source,Target,Weight,Type' + '\n')
#         countedges = collections.Counter(timedIdStaticEdges1)
#         alledges = [','.join([str(y) for y in x])+',undirected' for x in countedges.most_common()]
#         f.writelines('\n'.join(alledges))

#     with codecs.open(edgeWritePath+'/staticEdges'+Years+'lvl2.csv','w','utf8') as f:
#         f.write('Source,Target,Weight,Type' + '\n')
#         countedges = collections.Counter(timedIdStaticEdges2)
#         alledges = [','.join([str(y) for y in x])+',undirected' for x in countedges.most_common()]
#         f.writelines('\n'.join(alledges))

#     with codecs.open(edgeWritePath+'/staticEdges'+Years+'lvl3.csv','w','utf8') as f:
#         f.write('Source,Target,Weight,Type' + '\n')
#         countedges = collections.Counter(timedIdStaticEdges3)
#         alledges = [','.join([str(y) for y in x])+',undirected' for x in countedges.most_common()]
#         f.writelines('\n'.join(alledges))

#     with codecs.open(edgeWritePath+'/staticEdges'+Years+'lvlA.csv','w','utf8') as f:
#         f.write('Source,Target,Weight,Type' + '\n')
#         countedges = collections.Counter(timedIdStaticEdgesAll)
#         alledges = [','.join([str(y) for y in x])+',undirected' for x in countedges.most_common()]
#         f.writelines('\n'.join(alledges))

#     with codecs.open(statsWritePath+'/'+Years+'lvl1_unique_terms.tsv','w','utf8') as f:
#         countLvl1 = collections.Counter(alllvl1)
#         counted1 = [x[0] for x in countLvl1.most_common()]#['\t'.join([str(y) for y in x]) for x in countLvl1.most_common()]
#         f.writelines('\n'.join(counted1))
#     with codecs.open(statsWritePath+'/'+Years+'lvl1_unique_termsSizeSorted.tsv','w','utf8') as f:
#       f.writelines('\n'.join(sorted(set(alllvl1), key=lambda k: len(k))))

#     with codecs.open(statsWritePath+'/'+Years+'lvl2_unique_terms.tsv','w','utf8') as f:
#         countLvl2 = collections.Counter(alllvl2)
#         counted2 = [x[0] for x in countLvl2.most_common()]#['\t'.join([str(y) for y in x]) for x in countLvl2.most_common()]
#         f.writelines('\n'.join(counted2))
#     with codecs.open(statsWritePath+'/'+Years+'lvl2_unique_termsSizeSorted.tsv','w','utf8') as f:
#       f.writelines('\n'.join(sorted(set(alllvl2), key=lambda k: len(k))))

#     with codecs.open(statsWritePath+'/'+Years+'lvl3_unique_terms.tsv','w','utf8') as f:
#         countLvl3 = collections.Counter(alllvl3)
#         counted3 = [x[0] for x in countLvl3.most_common()]#['\t'.join([str(y) for y in x]) for x in countLvl3.most_common()]
#         f.writelines('\n'.join(counted3))
#     with codecs.open(statsWritePath+'/'+Years+'lvl3_unique_termsSizeSorted.tsv','w','utf8') as f:
#       f.writelines('\n'.join(sorted(set(alllvl3), key=lambda k: len(k))))

#     statement = '\n%s \nlvl1 has %s unique terms\nlvl2 has %s unique terms\nlvl3 has %s unique terms' %(Years,len(counted1),len(counted2),len(counted3))
#     print(statement)
#     fullstatement += statement
# with open(statsWritePath+'/simpleStats.txt','w') as f:
#     f.write(fullstatement)

print('finished')
time.sleep(3)

