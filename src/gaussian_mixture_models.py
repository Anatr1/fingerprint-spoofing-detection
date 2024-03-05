# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file runs the Gaussian Mixture Models on the dataset.

import utils
from plots import plotGMM
from PCA import PCA
from LDA import LDA
from CrossValidation import KFoldCrossValidation
from GMM import GMMFullCovariance, GMMNaiveBayes, GMMTiedCovariance

prior = 0.5
Cfn = 1
Cfp = 10

dataset, labels = utils.loadData("../data/Train.txt")

################# GMM #####################
number_of_components = 2

dcf_results = [[], [], [], [], [], [], []]
error_rate_results = [[], [], [], [], [], [], []]

for m in range(10, 6, -1):
    dataset_PCA = PCA(dataset, m)
    for Classifier in [GMMFullCovariance, GMMNaiveBayes, GMMTiedCovariance]:
        classifier = Classifier()
        print(classifier.id_string)
        print(f"PCA m={m}...")
        error_rate, error_rate_std_dev, dcf, dcf_std_dev = KFoldCrossValidation(classifier, dataset_PCA, labels, k=6, DIMENSIONS=m, components=number_of_components, prior=prior, Cfn=Cfn, Cfp=Cfp)
        error_rate_results[10 - m].append(round(error_rate, 3))
        dcf_results[10 - m].append(round(dcf, 3))

for m in range(9, 6, -1):
    dataset_LDA = LDA(dataset, labels, m)
    for Classifier in [GMMFullCovariance, GMMNaiveBayes, GMMTiedCovariance]:
        classifier = Classifier()
        print(classifier.id_string)
        print(f"LDA m={m}...")
        error_rate, error_rate_std_dev, dcf, dcf_std_dev = KFoldCrossValidation(classifier, dataset_LDA, labels, k=6, DIMENSIONS=m, components=number_of_components, prior=prior, Cfn=Cfn, Cfp=Cfp)
        error_rate_results[10 - (m + 4)].append(round(error_rate, 3))
        dcf_results[10 - (m + 4)].append(round(dcf, 3))

print(dcf_results)
print(error_rate_results)

print("               FC            NB            FTC")
for i in range(len(dcf_results)):
    print("m = " +  str(10 - i), end="\t")
    for j in range(len(dcf_results[i])):
        print(f"{dcf_results[i][j]} ({error_rate_results[i][j]}%)", end="\t")
    print()

plotGMM(dcf_results, number_of_components)
plotGMM(error_rate_results, number_of_components, error_rate=True)
