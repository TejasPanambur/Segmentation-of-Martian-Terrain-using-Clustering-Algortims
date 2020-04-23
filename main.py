# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:04:28 2019

@author: TEJAS
"""

import numpy as np
import matplotlib.pyplot as plt
from Functions import imgLibs , computeLibs,plotLibs

#################################################################################
'''Switches to Hyper Parameters To play with'''
colorSpace_s = 'rgb' # rgb, hsv, gray
decorrStretch_s = 1 #set 1 to perform decorelation stretch
plt_3d_s = 0 #set 1 to plot in 3d
kmeans_s = 1 #set 1 for performing kmeans
find_k_kmeans_s = 0 #To find optimal k bruteforce with k values of range
imageNum = 1 # 0,1,2,3 numbers for 4 images
clustering_alg = 0 #0 none,1 clustering is MeanShift, 2 is AgglomerativeClustering,3 is DBSCAN

##################################################################################
images = ['0153MR0008490000201265E01_DRLX','0172ML0009240000104879E01_DRLX','0172ML0009240340104913E01_DRLX','0270MR0011860360203259E01_DRLX']
best_tols = [None,None,None,10e-5] #Observed
norm_elbows = [6,4,6,4]  #Best K befor decorrelation
decor_elbows = [6,8,6,6] #Best K after decorrelation
hsv_elbows = [6,6,6,6]
if decorrStretch_s:
    k = decor_elbows[imageNum]
elif colorSpace_s=='hsv':
    k = hsv_elbows[imageNum]
else:
    k = norm_elbows[imageNum]
    

tol = best_tols[imageNum]
imageName = images[imageNum]

plt_str = imageName+'_'+colorSpace_s+'_'

img = imgLibs(imageName+'.npy',colorSpace_s).loadImg()
original_img = img
plotLibs().dispImg(img,save=0,title=plt_str)

if plt_3d_s:
    plotLibs().plot_3d(img)

if decorrStretch_s:
    plt_str += '_decorrStretch'
    img = computeLibs().decorrstretch(img,tol)
    plotLibs().dispImg(img,title=plt_str)    
    if plt_3d_s:
        plotLibs().plot_3d(img)
    
if find_k_kmeans_s:
    l,h = 2,16
    plt_str += '_bruteForce'+str((l,h))
    _,_,loss,reconstImg = computeLibs().kmeans(img,bruteforceRange=(l,h),bruteforce=True)
    plotLibs().dispKmeansBruteImg(reconstImg,l,plt_str)
    plotLibs().plotLoss(loss,plt_str)

if kmeans_s:
    plt_str += '_kmeans_clust'
    labels,n_clusters,loss,reconstImg = computeLibs().kmeans(img,k)
    plotLibs().dispSegment(img,labels,n_clusters,plt_str)
    plotLibs().dispImg(reconstImg,plt_str)
    if decorrStretch_s:
        plotLibs().dispSegment(original_img,labels,n_clusters,plt_str)


'''Mean shift is Computationally expensive and others have Space complexity leading to memory error'''

if clustering_alg==1:
    labels = computeLibs().meanShift(img)    
elif clustering_alg==2:
    labels = computeLibs().agglomerativeClustering(img)
elif  clustering_alg==3:
    labels = computeLibs().Dbscan(img)
    
    


    