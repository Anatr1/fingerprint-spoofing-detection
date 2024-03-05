# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file runs the Multivariate Gaussian Classifiers on the dataset.

import utils
from plots import plotMVG
from PCA import PCA
from LDA import LDA
from CrossValidation import KFoldCrossValidation
from MVG import GaussianClassifierFullCovariance, GaussianClassifierNaiveBayes, GaussianClassifierFullTiedCovariance, GaussianClassifierNaiveBayesTiedCovariance

prior = 0.5
Cfn = 1
Cfp = 10

dataset, labels = utils.loadData("../data/Train.txt")

################# MVG #####################
FC = GaussianClassifierFullCovariance()
NB = GaussianClassifierNaiveBayes()
FTC = GaussianClassifierFullTiedCovariance()
NBT = GaussianClassifierNaiveBayesTiedCovariance()

dcf_results = [[], [], [], [], [], [], []]
error_rate_results = [[], [], [], [], [], [], []]

for m in range(10, 6, -1):
    dataset_PCA = PCA(dataset, m)
    for classifier in [FC, NB, FTC, NBT]:
        error_rate, error_rate_std_dev, dcf, dcf_std_dev = KFoldCrossValidation(classifier, dataset_PCA, labels, k=6, DIMENSIONS=m, prior=prior, Cfn=Cfn, Cfp=Cfp)
        error_rate_results[10 - m].append(round(error_rate, 3))
        dcf_results[10 - m].append(round(dcf, 3))

for m in range(9, 6, -1):
    dataset_LDA = LDA(dataset, labels, m)
    for classifier in [FC, NB, FTC, NBT]:
        error_rate, error_rate_std_dev, dcf, dcf_std_dev = KFoldCrossValidation(classifier, dataset_LDA, labels, k=6, DIMENSIONS=m, prior=prior, Cfn=Cfn, Cfp=Cfp)
        error_rate_results[10 - (m + 4)].append(round(error_rate, 3))
        dcf_results[10 - (m + 4)].append(round(dcf, 3))

print(dcf_results)
print(error_rate_results)

print("               FC            DC              FCT           DCT")
for i in range(len(dcf_results)):
    print("m = " +  str(10 - i), end="\t")
    for j in range(len(dcf_results[i])):
        print(f"{dcf_results[i][j]} ({error_rate_results[i][j]}%)", end="\t")
    print()


plotMVG(dcf_results)
plotMVG(error_rate_results, error_rate=True)
