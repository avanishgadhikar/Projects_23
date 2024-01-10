#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:10:21 2023

@author: Avanish
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import eig

photo_data = plt.imread("Milky_way.jpg")
plt.figure(figsize = (10, 20))
plt.imshow(photo_data)
plt.title("Original Image")
plt.axis('off')
plt.show()

G = np.mean(photo_data, -1)
img = plt.imshow(G)
img.set_cmap("gray")
plt.title("Grayscale image")
plt.axis("off")
plt.show()

#Computing G@G_T
G_GT = G @ G.T
#perform eigenvalue deecomposition
eigenvalues, eigenvectors = np.linalg.eig(G_GT)

#Sort eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

#Compute the singular values and invert them
singular_values = np.sqrt(eigenvalues)
inv_singular_values = 1.0 / singular_values #TO generate Vt, mostlyGT_G will give complex eigenvalues. Hence using Av = lambda* u. Finding u in such a way

#Compute the U and VT matrices

U = eigenvectors
S = singular_values
S_G = np.diag((S))
V = G.T @ eigenvectors @np.diag(inv_singular_values)
VT = V.T

# U = Right singular eigvectors
# S = Sigma_matrix
# Vt = Left Singular matrix

ranks = [1,5, 15, 25, 50, 100, 500]

for i in range(np.shape(ranks)[0]):
    r = ranks[i]
    compressed_U = U[:, :r]
    compressed_VT = VT[:r, :]
    compressed_S = S_G[0:r, :r]
    
    # G = U_G @S_G @ V_G
    compressed_image = np.dot(np.dot(compressed_U, compressed_S), compressed_VT)
    
    
    img = plt.imshow(compressed_image)
    img.set_cmap("gray")
    plt.title("Compressed Grayscale image")
    plt.axis("off")
    plt.show()

    
                              
                              
                              
                              
                              
    
                              
                              