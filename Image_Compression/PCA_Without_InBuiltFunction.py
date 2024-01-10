# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:44:42 2023

@author: Teena Sharma
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.array([[4, 11],
                  [8, 4],
                  [13, 5],
                  [7, 14]])

plt.figure(figsize=(10, 5))
plt.scatter(data[:, 0], data[:, 1], label='Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()

mean_centered_data = data - np.mean(data, axis=0)

covariance_matrix = np.cov(mean_centered_data, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

k = 1
selected_eigenvectors = eigenvectors_sorted[:, :k]

reduced_data = mean_centered_data.dot(selected_eigenvectors)

# Get the principal component vector
principal_component = selected_eigenvectors[:, 0]

# Plot the original data and the reduced data
plt.figure(figsize=(10, 5))
plt.scatter(data[:, 0], data[:, 1], label='Original Data')
plt.scatter(reduced_data, np.zeros_like(reduced_data), 
            color='red', label='Reduced Data')
plt.quiver(np.mean(data[:, 0]), np.mean(data[:, 1]),
           principal_component[0] * 5, 
           principal_component[1] * 5,
           angles='xy', scale_units='xy', scale=1, 
           color='green', 
           label='Principal Component Vector')
plt.title('PCA: Original Data vs. Reduced Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2 / Reduced Dimension')
plt.legend()
plt.grid()
plt.show()
