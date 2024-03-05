# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file runs the Quadratic Logistic Regression on the dataset.


import numpy as np
import utils
from PCA import PCA
from LDA import LDA
from CrossValidation import KFoldCrossValidation
from plots import plotLambdaQLR, plotQLR
from LR import QuadraticLogisticRegression

prior = 0.5
Cfn = 1
Cfp = 10

dataset, labels = utils.loadData("../data/Train.txt")

################# QLR #####################
possible_lambdas = np.logspace(-6, 2, num=15)
values_for_lambda = []
best_lambda = 1e-6
best_minDCF = 1.0
best_m = 10
best_error = 100
preprocessing = "PCA"

dcf_results = []
error_rate_results = []

for m in range(10, 8, -1):
    dataset_PCA = PCA(dataset, m)
    current_best_dcf = 1.0
    current_best_error = 100
    for lambda_ in possible_lambdas:
        classifier = QuadraticLogisticRegression()
        print(f"PCA m = {m}, λ = {lambda_}")
        error_rate, error_rate_std_dev, dcf, dcf_std_dev = KFoldCrossValidation(classifier, dataset_PCA, labels, k=6,  DIMENSIONS=m, prior=prior, Cfn=Cfn, Cfp=Cfp, lambda_=lambda_)
        values_for_lambda.append(dcf) 
        if dcf < current_best_dcf:
            current_best_dcf = dcf
        if error_rate < current_best_error:
            current_best_error = error_rate         
        if dcf < best_minDCF:
            best_lambda = lambda_
            best_minDCF = dcf
            best_m = m
            preprocessing = "PCA"
        if error_rate < best_error:
            best_error = error_rate
        print()
    error_rate_results.append(current_best_error)
    dcf_results.append(current_best_dcf)
    print(dcf_results)
    print(error_rate_results)
    print()

for m in range(9, 8, -1):
    dataset_LDA = LDA(dataset, labels, m)
    current_best_dcf = 1.0
    current_best_error = 100
    for lambda_ in possible_lambdas:
        classifier = QuadraticLogisticRegression()
        print(f"LDA m = {m}, λ = {lambda_}")
        error_rate, error_rate_std_dev, dcf, dcf_std_dev = KFoldCrossValidation(classifier, dataset_LDA, labels, k=6,  DIMENSIONS=m, prior=prior, Cfn=Cfn, Cfp=Cfp, lambda_=lambda_)
        values_for_lambda.append(dcf)
        if dcf < current_best_dcf:
            current_best_dcf = dcf
        if error_rate < current_best_error:
            current_best_error = error_rate        
        if dcf < best_minDCF:
            best_lambda = lambda_
            best_minDCF = dcf
            best_m = m
            preprocessing = "LDA"
        if error_rate < best_error:
            best_error = error_rate
        print()
    error_rate_results.append(current_best_error)
    dcf_results.append(current_best_dcf)
    print(dcf_results)
    print(error_rate_results)
    print()

print(values_for_lambda)
print(dcf_results)
print(error_rate_results)
print(best_lambda)
print(best_minDCF)
print(best_m)
print(preprocessing)
plotLambdaQLR(possible_lambdas, values_for_lambda)
plotQLR(dcf_results)
plotQLR(error_rate_results, error_rate=True)
