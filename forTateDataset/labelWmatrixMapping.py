'''
    Assign the labels created by Wmatrix and cleaned by Sandor
'''
print('Assign the labels created by Wmatrix and cleaned by Sandor')
#--------------------------------------------
import pickle, time, pprint,sys, glob
from prettytable import PrettyTable
#--------------------------------------------

LVLs = ['lvl1','lvl2','lvl3','lvlA'] 
yearPeriods = ['1800s','2000s']

try:
    termLabelDict = pickle.load(open('./data/artworks_verification_labels/WmatrixLabelDict.pck','rb'))
except:
    termLabelDict = {}
    pass

files = glob.glob('./data/artworks_verification_labels/*.txt')
# for lvl in LVLs:
    # for years in yearPeriods:
        # print([lvl,years])
for tmpfile in files:
    with open(tmpfile,'r') as f:
        next(f)
        for xy in f:
            flag = False
            try:
                x,y,z = xy.strip().split('\t')
            except:
                print('error----------')
                print(xy)
                sys.exit
            if x in termLabelDict:
                flag = True
                test = termLabelDict[x]
            termLabelDict[x] = {'code':y,'label':z}
            if flag and test != termLabelDict[x]:
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%\nTerm is %s'%x)
                pprint.pprint(test)
                print('------------------------------------duplicate')
                pprint.pprint(termLabelDict[x])
                xa=input('gimme')

allTerms = list(termLabelDict.keys())
allTerms.sort()
pretty = PrettyTable(["Tate Term", "WMX Code", "WMX category"])
for x in allTerms:
    try:
        pretty.add_row([x,termLabelDict[x]['code'],termLabelDict[x]['label']])
    except:
        pretty.add_row([x,' ',' '])
        pass
with open('./data/artworks_verification_labels/WmatrixLabelDict.tsv','w') as f:
    f.write(str(pretty))

# pprint.pprint(termLabelDict)
pickle.dump(termLabelDict,open('./data/artworks_verification_labels/WmatrixLabelDict.pck','wb'), protocol = 2)
