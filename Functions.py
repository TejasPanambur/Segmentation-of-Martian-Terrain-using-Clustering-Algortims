# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:25:37 2019

@author: TEJAS
"""
import numpy as np
from functools import reduce
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering,DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import cv2
import matplotlib.pyplot as plt

class computeLibs:
    
    def decorrstretch(self,A, tol=None):
        """
        Apply decorrelation stretch to image
        Arguments:
        A   -- image in cv2/numpy.array format
        tol -- upper and lower limit of contrast stretching
        """
    
        # save the original shape
        orig_shape = A.shape
        # reshape the image
        #         B G R
        # pixel 1 .
        # pixel 2   .
        #  . . .      .
        A = A.reshape((-1,3)).astype(np.float)
        # covariance matrix of A
        cov = np.cov(A.T)
        # source and target sigma
        sigma = np.diag(np.sqrt(cov.diagonal()))
        # eigen decomposition of covariance matrix
        eigval, V = np.linalg.eig(cov)
        # stretch matrix
        S = np.diag(1/np.sqrt(eigval))
        # compute mean of each color
        mean = np.mean(A, axis=0)
        # substract the mean from image
        A -= mean
        # compute the transformation matrix
        T = reduce(np.dot, [sigma, V, S, V.T])
        # compute offset 
        offset = mean - np.dot(mean, T)
        # transform the image
        A = np.dot(A, T)
        # add the mean and offset
        A += mean + offset
        # restore original shape
        B = A.reshape(orig_shape)
        # for each color...
        for b in range(3):
            # apply contrast stretching if requested
            if tol:
                # find lower and upper limit for contrast stretching
                low, high = np.percentile(B[:,:,b], 100*tol), np.percentile(B[:,:,b], 100-100*tol)
                B[B<low] = low
                B[B>high] = high
            # ...rescale the color values to 0..255
            B[:,:,b] = 1 * (B[:,:,b] - B[:,:,b].min())/(B[:,:,b].max() - B[:,:,b].min())
        # return it as uint8 (byte) image
        return np.asarray(B,dtype='float32')
    
    def __MSE(self,Im1, Im2):
    	# computes error
    	Diff_Im = Im2-Im1
    	Diff_Im = np.power(Diff_Im, 2)
    	Diff_Im = np.sum(Diff_Im, axis=2)
    	Diff_Im = np.sqrt(Diff_Im)
    	sum_diff = np.sum(np.sum(Diff_Im))
    	avg_error = sum_diff / float(Im1.shape[0]*Im2.shape[1])
    	return avg_error
    
    def __KmeansHelper(self,img,no_of_clusters):
        imgShapeLen = len(img.shape)
        if imgShapeLen==3:
            H,W,C =  img.shape
            k_img = img.reshape(H*W,3)
        else:
            H,W,C,X,Y = img.shape 
            k_img = img.reshape(H*W*X*Y,3)
        
        Kmean = KMeans(n_clusters=no_of_clusters)
        Kmean.fit(k_img)
        kmean_clusters = np.asarray(Kmean.cluster_centers_,dtype=np.float32)
        if imgShapeLen ==3:
            reconstructedImg = kmean_clusters[Kmean.labels_,:].reshape(H,W,C)
        else:
            reconstructedImg = kmean_clusters[Kmean.labels_,:].reshape(H,W,C,X,Y)
        loss = self.__MSE(img,reconstructedImg)
        if imgShapeLen==3:
            labels = Kmean.labels_.reshape(H,W,1)
        else:
            labels = Kmean.labels_.reshape(H,W,X,Y,1)
        return labels,loss,reconstructedImg
    
    def kmeans(self,img,no_of_clusters=6,bruteforceRange=(0,12),bruteforce=False):
        labels = None        
        if bruteforce:
            l,h = bruteforceRange
            loss = []
            reconstructedImg = []
            for i in range(l,h):
                print('Starting Clustering with'+str(i)+'centers')
                _,l,reconstImg = self.__KmeansHelper(img,i)                               
                loss.append(l)
                reconstructedImg.append(reconstImg) 
        else:
            labels,loss,reconstructedImg = self.__KmeansHelper(img,no_of_clusters)   
        return labels,no_of_clusters,loss,reconstructedImg                         

    def meanShift(self,img):
        img = img.reshape(-1,3)
        clustering = MeanShift(bandwidth=10,min_bin_freq=20,bin_seeding=True).fit(img)
        labels = clustering.labels_
        return labels
    
    def agglomerativeClustering(self,img,n_clusters = 6):
        img = img.reshape(-1,3)
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(img)
        labels = clustering.labels_
        return labels
    
    def Dbscan(self,img,eps=0.3,min_samples=5):
        img = img.reshape(-1,3)
        clustering = DBSCAN(eps, min_samples).fit(img)
        labels = clustering.labels_
        return labels
        
class plotLibs:
    
    def dispImg(self,img,k=0,title='image',save=1):
        plt.figure(figsize=(10,10))
        plt.title(title)
        plt.imshow(img)
        if save:
            if k:
                plt.savefig('Plots/Kmeans/k_means_'+str(k)+'.jpg')
            else:
                plt.savefig('Plots/Others/'+title+'.jpg')
        plt.show()
    
    def plot_3d(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(img)
        r,g,b =  r.flatten(), g.flatten(), b.flatten()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(r, g, b)
        plt.show()
        
    def __segments(self,img,labels,center):
        labels = np.array(labels,dtype='float32')
        labels[labels!=center]= np.nan
        labels[labels==center]=1
        labels[labels==np.nan] = 0
        return img*labels
        
    def dispSegment(self,img,labels,number_of_clusters,name):
        M,N = number_of_clusters//3,3
        fig, axs = plt.subplots(M,N, figsize=(60, 60), facecolor='w', edgecolor='k',squeeze=True)
        fig.subplots_adjust(hspace = 0.1, wspace=.01)
        axs = axs.ravel()
        for i in range(number_of_clusters):
            segment = self.__segments(img,labels,i)
            axs[i].imshow(segment)
            axs[i].set_title('segment'+str(i))  
        plt.savefig('Plots/Segments/'+name+str(number_of_clusters)+'.jpg')
        
    def dispKmeansBruteImg(self,reconstructedImg,l,plt_name):
        M,N = len(reconstructedImg)//2,2
        fig, axs = plt.subplots(M,N, figsize=(60, 60), facecolor='w', edgecolor='k',squeeze=True)
        fig.subplots_adjust(hspace = 0.1, wspace=.01)
        axs = axs.ravel()
        for i in range(len(reconstructedImg)):
            axs[i].imshow(reconstructedImg[i])
            axs[i].set_title('K_'+str(i+l))  
        plt.savefig('Plots/Kmeans/'+plt_name+'.jpg')
        plt.show()
        
    def plotLoss(self,Loss,plt_name):
        plt.plot(Loss)
        plt.savefig('Plots/Kmeans/Loss_'+plt_name+'.jpg') 
        plt.show() 
        
                    
    
class imgLibs:
    
    def __init__(self,imgName,clrSpace='rgb'):
        self.img = np.load(imgName)
        self.imgShape = self.img.shape
        self.clrSpace = clrSpace
    
    def loadImg(self):
        self.img[np.isnan(self.img)] = 0
        if self.clrSpace =='rgb':
            return self.img
        elif self.clrSpace == 'hsv':
            return cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        elif self.clrSpace == 'gray':
            return cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        
        

        