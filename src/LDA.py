# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file contains the functions to perform Linear Discriminant Analysis.

import numpy as np
import scipy.linalg as sc
import utils

def computeBetweenClassCovarianceMatrix(dataset, labels, SVM=False):
    # Compute mean over columns of the dataset matrix
    mu = dataset.mean(axis=1)

    # Reshape the 1-D array mu to a column vector 4x1
    mu = utils.mcol(mu)

    # Compute classes means over columns of the dataset matrix
    if not SVM:
        mu0 = dataset[:, labels == 0].mean(axis=1)
    else:
        mu0 = dataset[:, labels == -1].mean(axis=1)
    mu1 = dataset[:, labels == 1].mean(axis=1)

    # Reshape all of them as column vectors
    mu0 = utils.mcol(mu0)
    mu1 = utils.mcol(mu1)

    # Count number of elements in each class
    if not SVM:
        n0 = dataset[:, labels == 0].shape[1]
    else:
        n0 = dataset[:, labels == -1].shape[1]
    n1 = dataset[:, labels == 1].shape[1]

    result = (1 / (n0 + n1)) * (
        (n0 * np.dot(mu0 - mu, (mu0 - mu).T)) + (n1 * np.dot(mu1 - mu, (mu1 - mu).T))
    )

    return result


def computeWithinClassCovarianceMatrix(dataset, labels, SVM=False):
    # Compute classes means over columns of the dataset matrix
    if not SVM:
        mu0 = dataset[:, labels == 0].mean(axis=1)
    else:
        mu0 = dataset[:, labels == -1].mean(axis=1)
    mu1 = dataset[:, labels == 1].mean(axis=1)

    # Reshape all of them as column vectors
    mu0 = utils.mcol(mu0)
    mu1 = utils.mcol(mu1)

    # Count number of elements in each class
    if not SVM:
        n0 = dataset[:, labels == 0].shape[1]
    else:
        n0 = dataset[:, labels == -1].shape[1]
    n1 = dataset[:, labels == 1].shape[1]

    # Compute within covariance matrix for each class
    Sw0 = (1 / n0) * np.dot(
        dataset[:, labels == 0] - mu0, (dataset[:, labels == 0] - mu0).T
    )
    Sw1 = (1 / n1) * np.dot(
        dataset[:, labels == 1] - mu1, (dataset[:, labels == 1] - mu1).T
    )

    result = (1 / (n0 + n1)) * (n0 * Sw0 + n1 * Sw1)

    return result


def computeLDA(SB, SW, dataset, m):
    # Solve the generalized eigenvalue problem
    _, U = sc.eigh(SB, SW)

    # Compute W matrix from U
    W = U[:, ::-1][:, 0:m]

    # LDA projection matrix
    projection_matrix = np.dot(W.T, dataset)

    return projection_matrix


def LDA(dataset, labels, m, SVM=False):
    SB = computeBetweenClassCovarianceMatrix(dataset, labels, SVM)
    SW = computeWithinClassCovarianceMatrix(dataset, labels, SVM)

    return computeLDA(SB, SW, dataset, m)
