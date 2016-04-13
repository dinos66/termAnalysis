'''
    Scrape summary, caption and extensive summary from Tate artworks from confirmed urls
'''
print('Scrape summary, caption and extensive summary from Tate artworks from confirmed urls')
#---------------------------------------
from lxml import html
import requests, webbrowser
import json, time, os, codecs, pickle
from os import walk
from nltk.tokenize import RegexpTokenizer
#---------------------------------------
print(time.asctime( time.localtime(time.time()) ))
tokenizer = RegexpTokenizer('\s+', gaps=True)

targetpath = './data/artworks'
tsvWritePath = './data/artworks_Summaries'

t = time.time()

try:
    fullDict = pickle.load(open('./data/artworks_tmp/summaryCaptionDict.pck','rb'))
    print('loaded dictionary in the works')
except:    
    fullDict = {'fileIds':set()}
    fullDict['allCount'], fullDict['summaryCount'], fullDict['captionCount'] = 0, 0, 0
    print('new dictionary')
    pass

fullDict['allCount'], fullDict['summaryCount'], fullDict['captionCount'] = 0, 0, 0

print('Already processed %s fileIds' %len(fullDict['fileIds']))

try:
    existingsummaryUrls = pickle.load(open('./data/artworks_tmp/existingsummaryUrls.pck','rb'))
    print('using processed existingsummaryUrls')
except:
    existingsummaryUrls = set()
    pass
try:
    existingcaptionUrls = pickle.load(open('./data/artworks_tmp/existingcaptionUrls.pck','rb'))
    print('using processed existingcaptionUrls')
except:
    existingcaptionUrls = set()
    pass
print('existingsummaryUrls are %s and existingcaptionUrls are %s' %(len(existingsummaryUrls), len(existingcaptionUrls)))

for dirname, dirnames, filenames in walk(targetpath):
    for filename in filenames:
        fileId = filename[:-5]
        fullDict['allCount']+=1
        captionFlag, summaryFlag = False,False
        if fileId in fullDict and 'caption' in fullDict[fileId] and 'summary' in fullDict[fileId]:
            fullDict['captionCount'] += 1
            fullDict['summaryCount'] += 1
            continue
        if fileId in fullDict and 'caption' in fullDict[fileId]:
            fullDict['captionCount'] += 1
            captionFlag = True
        if fileId in fullDict and 'summary' in fullDict[fileId]:
            fullDict['summaryCount'] += 1
            summaryFlag = True
        if fileId not in fullDict['fileIds']:
            filepath = '/'.join([dirname,filename])
            fileopen = open(filepath).read()
            jsonfile = json.loads(fileopen)
            mainUrl = jsonfile['url']
            displayUrl = mainUrl+'/text-display-caption'
            summaryUrl = mainUrl+'/text-summary'

            if not captionFlag and displayUrl in existingcaptionUrls:
                try:
                    newpage = requests.get(displayUrl)
                    tree = html.fromstring(newpage.content)
                    textCaption = tree.xpath('//div[@class="texts_content"]')
                    fullCaption = ' '.join(tokenizer.tokenize(' '.join([x.text_content() for x in textCaption])))
    ##                print('caption is: ')
    ##                print(fullCaption)          
                    if fileId in fullDict:  
                        fullDict[fileId]['caption'] = fullCaption
                        fullDict[fileId]['displayUrl'] = displayUrl     
                    else:
                        fullDict[fileId] = {'caption':fullCaption,'displayUrl':displayUrl}
                    fullDict['captionCount'] += 1
        ##            webbrowser.open_new_tab(displayUrl)
                except:
                    webbrowser.open_new_tab(displayUrl)
                    pass
                
            if not summaryFlag and summaryUrl in existingsummaryUrls:
##                webbrowser.open_new_tab(summaryUrl)
##                xa=input('continue?')
                try:
                    newpage = requests.get(summaryUrl)
                    tree = html.fromstring(newpage.content)
                    textSummary = tree.xpath('//div[@class="texts_content"]')   
                    tmpSummary = ' '.join([x.text_content() for x in textSummary])
                    if 'Further reading:' in tmpSummary:
                        tmpSummary = tmpSummary[:tmpSummary.index('Further reading:')]
                    elif 'Further reading' in tmpSummary:
                        tmpSummary = tmpSummary[:tmpSummary.index('Further reading')]
                    fullSummary = ' '.join(tokenizer.tokenize(tmpSummary))         
                    # print('summary is: ')            
                    # print(fullSummary)
                    if fileId in fullDict:  
                        fullDict[fileId]['summary'] = fullSummary
                        fullDict[fileId]['summaryUrl'] = summaryUrl                    
                    else:
                        fullDict[fileId] = {'summary':fullSummary,'summaryUrl':summaryUrl}

                    fullDict['summaryCount'] += 1
                    # webbrowser.open_new_tab(summaryUrl)
                    # xa=input('continue?')
                except:
                    webbrowser.open_new_tab(summaryUrl)
                    pass
                
        if not fullDict['allCount']%1000:
            pickle.dump(fullDict,open('./data/artworks_tmp/summaryCaptionDict.pck','wb'), protocol = 2)
            print('@@@@@ Just passed file number '+str(fullDict['allCount'])+' at '+time.strftime("%H:%M||%d/%m "))
            time.sleep(10)
print('allcount: %s  summaryCount: %s  captionCount: %s' %(fullDict['allCount'], fullDict['summaryCount'], fullDict['captionCount']))
pickle.dump(fullDict,open('./data/artworks_tmp/summaryCaptionDict.pck','wb'), protocol = 2)

elapsed = time.time() - t
print('Elapsed: %.2f seconds' % elapsed)


