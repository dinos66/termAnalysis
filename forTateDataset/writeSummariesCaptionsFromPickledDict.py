'''write summaries and captions from pickled dict'''
print('write summaries and captions from pickled dict')
#---------------------------------------
import json, time, os, codecs, pickle
#---------------------------------------
print(time.asctime( time.localtime(time.time()) ))

fullDict = pickle.load(open('./data/artworks_tmp/summaryCaptionDict.pck','rb'))

totalArt = fullDict['fileIds']

print(len(totalArt))
print(fullDict['allCount'])
print(fullDict['summaryCount'])
print(fullDict['captionCount'])

fileIds = list(fullDict.keys())
fileIds.remove('fileIds')
fileIds.remove('allCount')
fileIds.remove('summaryCount')
fileIds.remove('captionCount')

with codecs.open('./data/artworks_tmp/captions.tsv','w','utf8') as fc: 
	with codecs.open('./data/artworks_tmp/summaries.tsv','w','utf8') as fs:
		for Id in fileIds:
			if 'caption' in fullDict[Id] and fullDict[Id]['caption']:
				fc.write(Id+'\t'+fullDict[Id]['displayUrl']+'\t'+fullDict[Id]['caption']+'\n')
			if 'summary' in fullDict[Id] and fullDict[Id]['summary']:
				fs.write(Id+'\t'+fullDict[Id]['summaryUrl']+'\t'+fullDict[Id]['summary']+'\n')
