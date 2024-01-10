#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:19:23 2023

@author: Avanish
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.image import imread
import math

#Load a Sample image
image = imread("Milky_way.jpg")
image = np.mean(image, -1)
data = image / 255.0 #Normalize pixel values to [0, 1]

def psnr(original, compressed):
    mse = np.mean((original - compressed)**2)
    if mse == 0:
        return 100 #PSNR is infinity when mse is 0
    max_pixel = 255.0
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse))
    
    return psnr_value

psnr_values_PCA = []
psnr_values_SVD = []
ranks = range(5, 510, 10)
for r in ranks:
    pca = PCA(n_components = r)
    compressed_data = pca.fit_transform(data)
    reconstructed_data = pca.inverse_transform(compressed_data)
    #Reshape the compressed image back to original shape
    compressed_image_PCA = reconstructed_data.reshape(image.shape)
    #Perform SVD
    U, S, VT = np.linalg.svd(data, full_matrices = True)
    Sigma_matrix = np.zeros((np.shape(data)[0], np.shape(data)[1]))
    min_dim = min(np.shape(data)[0], np.shape(data)[1])
    for i in range(min_dim):
        Sigma_matrix[i, i] = S[i]
        
    S = Sigma_matrix
    compressed_image_SVD =  U[:, :r]@S[0:r, :r]@VT[:r, :]
    
    #Calculate and display PSNR
    psnr_values_PCA = psnr(data, compressed_image_PCA)
    psnr_values_SVD = psnr(data, compressed_image_SVD)
    print(f"PSNR for rank {r} : PCA - {psnr_values_PCA:.2f}dB, SVD - {psnr_values_SVD:.2f}dB")
    psnr_values_PCA.append(psnr_values_PCA)
    psnr_values_SVD.append(psnr_values_SVD)
    
    #Display original and compressed image using PCA and SVD
    plt.figure(figsize = (10, 5))
    plt.subplot(1, 3, 1)
    img = plt.imshow(data)
    img.set_cmap('gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title
    
        
        
        
        
        
        
        