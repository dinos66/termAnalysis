'''
    Fix the labels created by Wmatrix
'''
print('Fix the labels created by Wmatrix')
#--------------------------------------------
#run create_Info_Files.py before running this
#--------------------------------------------
import pickle, time, pprint

LVLs = ['lvl1','lvl2','lvl3'] 
heatmapFonts = [12,7,6]
yearPeriods = ['1800s','2000s']

tagSetDict = {}
with open('./data/artworks_verification_labels/labelMapping.txt','r') as f:
    next(f)
    for xy in f:
        x,y = xy.strip().split('\t')
        tagSetDict[x] = y

termLabelDict = {}
for lvl in LVLs:
    for years in yearPeriods:
        with open('./data/artworks_verification_labels/'+years+lvl+'.txt','r') as f:
            next(f)
            tmpTerms = []
            for xy in f:
                x,y = xy.strip().split('\t')
                termLabelDict[x] = y
                tmpTerms.append(x)
                # try:
                #     termLabelDict[x] = tagSetDict[y]
                # except:
                #     try:
                #         termLabelDict[x] = tagSetDict[y[:-1]]
                #     except:
                #         try:
                #             termLabelDict[x] = tagSetDict[y[:-2]]
                #         except:
                #             print(y)
        with open('./data/artworks_verification_labels/tateCodeCategoryList'+years+lvl+'.txt','w') as f:
            tmpTerms.sort()
            for x in tmpTerms:
                try:
                    f.write('\t'.join([x,termLabelDict[x],tagSetDict[termLabelDict[x]]])+'\n')
                except:
                    f.write('\t'.join([x,' ',' '])+'\n')
                    pass

allTerms = list(termLabelDict.keys())
allTerms.sort()
with open('./data/artworks_verification_labels/tateCodeCategoryList.txt','w') as f:
    for x in allTerms:
        try:
            f.write('\t'.join([x,termLabelDict[x],tagSetDict[termLabelDict[x]]])+'\n')
        except:
            f.write('\t'.join([x,' ',' '])+'\n')
            pass
# pprint.pprint(termLabelDict)
pickle.dump(termLabelDict,open('./data/artworks_verification_labels/WmatrixLabelDict.pck','wb'), protocol = 2)
