# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file contains the functions to perform Principal Component Analysis.

import numpy as np
import utils

def computePCA(covariance_matrix, dataset, m):
    # Use linalg.eigh to get eigenvalues and eigenvectors of C,
    # then computes its principal components and projection matrix
    _, eigenvectors = np.linalg.eigh(covariance_matrix)
    principal_components = eigenvectors[:, ::-1][:, 0:m]

    projection_matrix = np.dot(principal_components.T, dataset)

    return projection_matrix

def getEigen(dataset):
    # Center the data
    centered_dataset = dataset - utils.mcol(dataset.mean(axis=1))

    # Compute covariance matrix
    covariance_matrix = (1 / centered_dataset.shape[1]) * (
        np.dot(centered_dataset, centered_dataset.T)
    )

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors

def PCA(dataset, m):
    if m >= dataset.shape[0]:
        return dataset
    
    # First we need to center the data
    centered_dataset = dataset - utils.mcol(dataset.mean(axis=1))

    # Compute covariance matrix
    covariance_matrix = (1 / centered_dataset.shape[1]) * (
        np.dot(centered_dataset, centered_dataset.T)
    )

    return computePCA(covariance_matrix, dataset, m)
