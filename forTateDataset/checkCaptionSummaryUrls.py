#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:GDD_parser
# Purpose:       This .py file extracts urls from json twitter files.
#
# Required libs: python-dateutil,pyparsing,numpy,matplotlib,networkx
# Author:        konkonst
#
# Created:       20/08/2013
# Copyright:     (c) ITI (CERTH) 2013
# Licence:       <apache licence 2.0>
#-------------------------------------------------------------------------------
import json,codecs,os,glob,time, pickle, collections, requests
import concurrent.futures
from os import walk

urlsInParallel = 10

targetpath = './data/artworks'
session = requests.Session()

def load_url(url, timeout):
    try:
        resp = session.head(url, allow_redirects=True, timeout = timeout)
        trueUrl = resp.url
    except:
        trueUrl = 'else'
        pass
    return trueUrl
    
print('check summary and caption urls')

try:
    potentialsummaryUrls = pickle.load(open('./data/artworks_tmp/potentialsummaryUrls.pck','rb'))
    potentialcaptionUrls = pickle.load(open('./data/artworks_tmp/potentialcaptionUrls.pck','rb'))
    print('using processed potential summaryUrls')
except:
    potentialsummaryUrls, potentialcaptionUrls = [], []
    for dirname, dirnames, filenames in walk(targetpath):
        for filename in filenames:
            fileId = filename[:-5]
            filepath = '/'.join([dirname,filename])
            fileopen = open(filepath).read()
            jsonfile = json.loads(fileopen)        
            mainUrl = jsonfile['url']
            potentialsummaryUrls.append(mainUrl+'/text-summary')
            potentialcaptionUrls.append(mainUrl+'/text-display-caption')
print('potentialsummaryUrls are %s and potentialcaptionUrls are %s' %(len(potentialsummaryUrls), len(potentialcaptionUrls)))

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

t = time.time()
for i in range(3):
    if i:
        print('Repassing to ensure full unshortening')    
    tmpsummaryUrls = [x for x in potentialsummaryUrls if x not in existingsummaryUrls]
    summaryShorts = [tmpsummaryUrls[x:x+urlsInParallel] for x in range(0, len(tmpsummaryUrls), urlsInParallel)]
    tmpcaptionUrls = [x for x in potentialcaptionUrls if x not in existingcaptionUrls]
    captionShorts = [tmpcaptionUrls[x:x+urlsInParallel] for x in range(0, len(tmpcaptionUrls), urlsInParallel)]

    summaryurlLength = len(summaryShorts)
    print('There are '+str(summaryurlLength)+' batches of '+str(urlsInParallel)+' summary urls')
    tssumm = int(summaryurlLength/urlsInParallel)

    captionurlLength = len(captionShorts)
    print('There are '+str(captionurlLength)+' batches of '+str(urlsInParallel)+' caption urls')
    tscapt = int(captionurlLength/urlsInParallel)

    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
        for idx,tmpshorts in enumerate(summaryShorts):
            # Start the load operations and mark each future with its URL   
            future_to_url = {executor.submit(load_url, url, 10): url for url in tmpshorts}
            try:
                for future in concurrent.futures.as_completed(future_to_url, timeout=60):
                    thisUrl = future_to_url[future]
                    trueUrl = future.result()
                    if trueUrl and thisUrl==trueUrl:                    
                        existingsummaryUrls.add(trueUrl)
            except concurrent.futures._base.TimeoutError:
                pass
            if not idx%200:
                pickle.dump(existingsummaryUrls, open('./data/artworks_tmp/existingsummaryUrls.pck','wb'))
                print('@@@@@ Just passed batch '+str(idx)+' at '+time.strftime("%H:%M||%d/%m ")) 

    with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
        for idx,tmpshorts in enumerate(captionShorts):
            # Start the load operations and mark each future with its URL   
            future_to_url = {executor.submit(load_url, url, 10): url for url in tmpshorts}
            try:
                for future in concurrent.futures.as_completed(future_to_url, timeout=60):
                    thisUrl = future_to_url[future]
                    trueUrl = future.result()
                    if trueUrl and thisUrl==trueUrl:                    
                        existingcaptionUrls.add(trueUrl)
            except concurrent.futures._base.TimeoutError:
                pass
            if not idx%200:
                pickle.dump(existingcaptionUrls, open('./data/artworks_tmp/existingcaptionUrls.pck','wb'))
                print('@@@@@ Just passed batch '+str(idx)+' at '+time.strftime("%H:%M||%d/%m ")) 

    pickle.dump(existingsummaryUrls, open('./data/artworks_tmp/existingsummaryUrls.pck','wb'))
    pickle.dump(existingcaptionUrls, open('./data/artworks_tmp/existingcaptionUrls.pck','wb'))


    elapsed = time.time() - t
    print('Elapsed: %.2f seconds' % elapsed)
    t = time.time()

