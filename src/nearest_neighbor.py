# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file runs the Nearest Neighbors Regression on the dataset.

import utils
import numpy as np
from PCA import PCA
from LDA import LDA
from plots import plotKNN   
from NNC import NearestNeighborClassifier, KNearestNeighborClassifier, WeightedKNearestNeighborClassifier
from CrossValidation import KFoldCrossValidation

prior = 0.5
Cfn = 1
Cfp = 10

dataset, labels = utils.loadData("../data/Train.txt")

################# KNN #####################
classifier = WeightedKNearestNeighborClassifier()

possible_ks = np.logspace(0, 2, num=20).astype(int)
possible_ks[possible_ks % 2 == 0] += 1
possible_ks = np.unique(possible_ks)

dcf_results = []
error_rate_results = []

for m in range(10, 6, -1):
    dataset_PCA = PCA(dataset, m)
    for k in possible_ks:
        classifier.k = k
        print(f"Fitting with PCA m = {m} and k = {k}")
        error_rate, error_rate_std_dev, dcf, dcf_std_dev = KFoldCrossValidation(classifier, dataset_PCA, labels, k=6, KNN=True, DIMENSIONS=m, prior=prior, Cfn=Cfn, Cfp=Cfp)
        error_rate_results.append(error_rate)
        dcf_results.append(dcf)
        print()
        

for m in range(9, 6, -1):
    dataset_LDA = LDA(dataset, labels, m)
    for k in possible_ks:
        classifier.k = k
        print(f"Fitting with LDA m = {m} and k = {k}")
        error_rate, error_rate_std_dev, dcf, dcf_std_dev = KFoldCrossValidation(classifier, dataset_LDA, labels, k=6, KNN=True, DIMENSIONS=m, prior=prior, Cfn=Cfn, Cfp=Cfp)
        error_rate_results.append(error_rate)
        dcf_results.append(dcf)
        print()

plotKNN(possible_ks, dcf_results)
plotKNN(possible_ks, error_rate_results, error_rate=True)
