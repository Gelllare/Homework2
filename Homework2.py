# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:16:23 2024

@author: Asus
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define your function F(x, y)
def F(x, y):
    return np.exp(1j*y) + 2*np.exp(-0.5*1j*y)*np.cos(np.sqrt(3)/2*x)


def a12(x, y):
    return F(x, y)

def a21(x, y):
    return np.conj(F(x, y))


def create_matrix(x, y):
    return np.array([[0, a12(x, y)],
                     [a21(x, y), 0]])



x_values = np.linspace(-np.pi, np.pi)
y_values = np.linspace(-np.pi, np.pi)
X, Y = np.meshgrid(x_values, y_values)


eigenvalues_real = np.zeros_like(X)
eigenvalues_imag = np.zeros_like(X)


for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        
        A = create_matrix(X[i,j], Y[i,j])
     
        eigvals, _ = np.linalg.eig(A)
        
        eigenvalues_real[i,j] = np.min(eigvals.real)  
        eigenvalues_imag[i,j] = np.min(eigvals.imag)  


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, eigenvalues_real, cmap='viridis', edgecolor='none')
ax.set_title('Real Part of Eigenvalues')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Real Eigenvalue')

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, eigenvalues_imag, cmap='viridis', edgecolor='none')
ax.set_title('Imaginary Part of Eigenvalues')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Imaginary Eigenvalue')

plt.show()