# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file runs the Linear Support Vector Machines on the dataset.

import numpy as np
import utils
from PCA import PCA
from LDA import LDA
from plots import plotLinearSVM
from SVM import LinearSVM

prior = 0.5
Cfn = 1
Cfp = 10

dataset, labels = utils.loadDataChangeLabels("../data/Train.txt")
(training_dataset, training_labels), (cross_validation_dataset, cross_validation_labels) = utils.singleFold(dataset, labels)

############### Linear SVM ###############
possible_Cs = np.logspace(-6, 2, num=10)
possible_Ks = [0.01, 0.1, 1, 10]

classifier = LinearSVM()
print(classifier.id_string)

dcf_results = []
err_results = []

for m in range(10, 7, -1):
    dataset_PCA = PCA(training_dataset, m)
    cross_validation_dataset_PCA = PCA(cross_validation_dataset, m)
    for K in possible_Ks:
        for C in possible_Cs:
            print(f"Training with PCA m = {m}, K = {K}, C = {C}")
            classifier.train(dataset_PCA, training_labels, C, K)
            dcf_results.append(round(utils.minDCF(classifier, cross_validation_dataset_PCA, cross_validation_labels, prior, Cfn, Cfp), 3))
            err_results.append(round(utils.computeErrorRate(classifier.predict(cross_validation_dataset_PCA), cross_validation_labels), 2))
            print(f"MinDCF: {dcf_results[-1]}, Error Rate: {err_results[-1]}")
        print(dcf_results)
        print(err_results)
        print()

for m in range(9, 7, -1):
    dataset_LDA = LDA(training_dataset, training_labels, m, SVM=True)
    cross_validation_dataset_LDA = LDA(cross_validation_dataset, cross_validation_labels, m, SVM=True)
    for K in possible_Ks:
        for C in possible_Cs:
            print(f"Training with LDA m = {m}, K = {K}, C = {C}")
            classifier.train(dataset_LDA, training_labels, C, K)
            dcf_results.append(round(utils.minDCF(classifier, cross_validation_dataset_LDA, cross_validation_labels, prior, Cfn, Cfp), 3))
            err_results.append(round(utils.computeErrorRate(classifier.predict(cross_validation_dataset_LDA), cross_validation_labels), 2))
            print(f"MinDCF: {dcf_results[-1]}, Error Rate: {err_results[-1]}")
        print(dcf_results)
        print(err_results)
        print()

plotLinearSVM(possible_Cs, dcf_results)
plotLinearSVM(possible_Cs, err_results, error_rate=True)