'''
    Scrape summary, caption and extensive summary from Tate artworks
'''
print('Scrape summary, caption and extensive summary from Tate artworks')
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
##fullDict['fileIds'] = set()
print('Already processed %s fileIds' %len(fullDict['fileIds']))


for dirname, dirnames, filenames in walk(targetpath):
    for filename in filenames:
        fileId = filename[:-5]
        fullDict['allCount']+=1
        if fileId in fullDict and 'caption' in fullDict[fileId] and 'summary' in fullDict[fileId]:
            fullDict['fileIds'].add(fileId)
            fullDict['captionCount'] += 1
            fullDict['summaryCount'] += 1
            continue
        if fileId in fullDict and 'caption' in fullDict[fileId]:
            fullDict['captionCount'] += 1
        if fileId in fullDict and 'summary' in fullDict[fileId]:
            fullDict['summaryCount'] += 1
        if fileId not in fullDict['fileIds']:
            flag1 = False
            flag2 = False
            filepath = '/'.join([dirname,filename])
            fileopen = open(filepath).read()
            jsonfile = json.loads(fileopen)
            mainUrl = jsonfile['url']
            displayUrl = mainUrl+'/text-display-caption'
            summaryUrl = mainUrl+'/text-summary'

            try:
                page = requests.get(mainUrl)
            except:
                continue

            if 'text-display-caption' in page.text and (fileId not in fullDict or 'caption' not in fullDict[fileId]):
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
                    flag1 = True
                    pass
                
            if 'text-summary' in page.text and (fileId not in fullDict or 'summary' not in fullDict[fileId]):
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
                    flag2 = True
                    pass
                
            if not flag1 and not flag2:
                fullDict['fileIds'].add(fileId)
##        xa=input('continue?')
        # if not fullDict['allCount']%100:
        #     print(fullDict['allCount'])
        if not fullDict['allCount']%1000:
            pickle.dump(fullDict,open('./data/artworks_tmp/summaryCaptionDict.pck','wb'), protocol = 2)
            print('@@@@@ Just passed file number '+str(fullDict['allCount'])+' at '+time.strftime("%H:%M||%d/%m "))
            time.sleep(10)
print('allcount: %s  summaryCount: %s  captionCount: %s' %(fullDict['allCount'], fullDict['summaryCount'], fullDict['captionCount']))
pickle.dump(fullDict,open('./data/artworks_tmp/summaryCaptionDict.pck','wb'), protocol = 2)

elapsed = time.time() - t
print('Elapsed: %.2f seconds' % elapsed)


