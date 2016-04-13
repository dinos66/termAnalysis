import json
from os import walk

allFilenames = []
for dirname, dirnames, filenames in walk('../artworks'):
        for filename in filenames:
                if filename.endswith('.json'):
                        allFilenames.append(filename)

print(len(allFilenames))
print(len(set(allFilenames)))
