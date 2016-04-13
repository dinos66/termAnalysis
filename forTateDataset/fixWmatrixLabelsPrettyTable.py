'''
    Fix the labels created by Wmatrix
'''
print('Fix the labels created by Wmatrix')
#--------------------------------------------
#run create_Info_Files.py before running this
#--------------------------------------------
import pickle, time, pprint
from prettytable import PrettyTable

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
        pretty = PrettyTable(["Tate Term", "WMX Code", "WMX category"])
        tmpTerms.sort()
        for x in tmpTerms:
            try:
                pretty.add_row([x,termLabelDict[x],tagSetDict[termLabelDict[x]]])
            except:
                pretty.add_row([x,' ',' '])
                pass
        with open('./data/artworks_verification_labels/tateCodeCategoryList'+years+lvl+'.txt','w') as f:
            f.write(str(pretty))

allTerms = list(termLabelDict.keys())
allTerms.sort()
pretty = PrettyTable(["Tate Term", "WMX Code", "WMX category"])
for x in allTerms:
    try:
        pretty.add_row([x,termLabelDict[x],tagSetDict[termLabelDict[x]]])
    except:
        pretty.add_row([x,' ',' '])
        pass
with open('./data/artworks_verification_labels/tateCodeCategoryList.txt','w') as f:
    f.write(str(pretty))

for lvl in LVLs:
    for years in yearPeriods:
        with open('./data/artworks_verification_labels/'+years+lvl+'_unique_persistent_terms.txt','r') as f:
            persistTerms = [x.strip() for x in f.readlines()]
            pretty = PrettyTable(["Tate Term", "WMX Code", "WMX category"])
            persistTerms.sort()
            for x in persistTerms:
                try:
                    pretty.add_row([x,termLabelDict[x],tagSetDict[termLabelDict[x]]])
                except:
                    pretty.add_row([x,' ',' '])
                    pass
            with open('./data/artworks_verification_labels/persistentTateCodeCategoryList'+years+lvl+'.txt','w') as f:
                f.write(str(pretty))
# pprint.pprint(termLabelDict)
pickle.dump(termLabelDict,open('./data/artworks_verification_labels/WmatrixLabelDict.pck','wb'), protocol = 2)
