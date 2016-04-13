'''
	Create adjacency matrices and analyse terms
'''
print('Create adjacency matrices and ESOMs')
#--------------------------------------------
#run create_Info_Files.py before running this
#--------------------------------------------
import igraph, glob, os, 

edgeReadPath = './data/artworks_edges'

try:
    edgeDict = pickle.load(open('./data/artworks_tmp/edgeDictStatic.pck','rb'))
except:
	files = glob.glob(edgeReadPath+'static*LVL2.txt')
	edgeDict = {}
	for filename in files:
		with open(filename, 'r') as f: