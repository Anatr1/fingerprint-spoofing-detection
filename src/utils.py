# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file contains utility functions.

import numpy as np
import json
import scipy
import sklearn.datasets as skds
from sklearn.metrics import roc_curve
from LR import LogisticRegression

def mcol(v):
    # 1-dim vectors -> column vectors.
    return v.reshape((v.size, 1))

def mrow(v):
    # 1-dim vectors -> row vectors.
    return v.reshape((1, v.size))

def load_iris():
    D, L = skds.load_iris()['data'].T, skds.load_iris()['target']
    return D, L

def load_iris_binary():
    D, L = skds.load_iris()['data'].T, skds.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = -1  # We assign label 0 to virginica (was label 2)
    return D, L
    
def logpdf_GMM(X, gmm):
    num_components = len(gmm)
    log_density = np.empty((num_components, X.shape[1]))
    
    for component_index in range(num_components):
        log_density[component_index, :] = getLogLikelihood(X, gmm[component_index][1], gmm[component_index][2])
        log_density[component_index, :] += np.log(gmm[component_index][0])

    log_density = scipy.special.logsumexp(log_density, axis=0)
    
    return log_density


def loadData(filename):
    # Extracts data and labels from a given text file.
    dataset = []
    labels = []

    with open(filename, "r") as file:
        for line in file:
            data = line.split(",")
            if data[0] != "\n":
                for i in range(len(data) - 1):
                    data[i] = float(data[i])

                data[-1] = int(data[-1].rstrip("\n"))

                dataset.append(mcol(np.array(data[0:-1])))
                labels.append(data[-1])

    dataset = np.hstack(dataset[:])
    labels = np.array(labels)

    return dataset, labels

def loadDataChangeLabels(filename):
    # Extracts data and labels from a given text file. Substitutes 0 labels with -1
    dataset = []
    labels = []

    with open(filename, "r") as file:
        for line in file:
            data = line.split(",")
            if data[0] != "\n":
                for i in range(len(data) - 1):
                    data[i] = float(data[i])

                data[-1] = int(data[-1].rstrip("\n"))

                dataset.append(mcol(np.array(data[0:-1])))
                labels.append(data[-1])

    dataset = np.hstack(dataset[:])
    labels = np.array(labels)
    labels[labels == 0] = -1

    return dataset, labels


def normalizeDataset(dataset):
    # normalizes data to a normal distribution
    mean = dataset.mean(axis=1)
    std_dev = dataset.std(axis=1)

    return (dataset - mcol(mean)) / mcol(std_dev)


def getLogLikelihood(x, mean, sigma):
    return (
        -(x.shape[0] / 2) * np.log(2 * np.pi)
        - (0.5) * (np.linalg.slogdet(sigma)[1])
        - (0.5)
        * np.multiply((np.dot((x - mean).T, np.linalg.inv(sigma))).T, (x - mean)).sum(
            axis=0
        )
    )

def computeAccuracy(predictions, labels):
    accurate_predictions = np.array(predictions == labels).sum()
    accuracy = accurate_predictions / labels.size * 100
    return accuracy


def computeErrorRate(predictions, labels):
    accuracy = computeAccuracy(predictions, labels)
    errorRate = 100 - accuracy
    return errorRate


def singleFold(dataset, labels, seed=0):
    np.random.seed(seed)
    n = int(dataset.shape[1] * 2 / 3)
    shuffled_dataset = np.random.permutation(dataset.shape[1])

    train_index = shuffled_dataset[0:n]
    validation_index = shuffled_dataset[n:]

    train_dataset = dataset[:, train_index]
    validation_dataset = dataset[:, validation_index]
    train_labels = labels[train_index]
    validation_labels = labels[validation_index]

    return (train_dataset, train_labels), (validation_dataset, validation_labels)


def confusionMatrix(predictions, labels, dim):
    confusion_matrix = np.zeros((dim, dim)).astype(int)
    for i in range(labels.size):
        label = labels[i]
        if label == -1:
            label = 0
        confusion_matrix[predictions[i], label] += 1
    return confusion_matrix


def optimalBayesConfusionMatrix(scores, labels, threshold):
    predictions = (scores > threshold).astype(int)
    confusion_matrix = confusionMatrix(predictions, labels, 2)
    return confusion_matrix


def computeFPRTPR(confusion_matrix):
    # Compute FNR and FPR
    FNR = confusion_matrix[0][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])
    TPR = 1-FNR
    FPR = confusion_matrix[1][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])
    return (FPR, TPR)


def computeDCF(pi, Cfn, Cfp, confusion_matrix):
    # Computes False Negative Rate and False Positive Rate
    FNR = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    FPR = confusion_matrix[1][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])

    # Computes the DCF
    return pi * Cfn * FNR + (1 - pi) * Cfp * FPR


def computeNormalizedDCF(pi, Cfn, Cfp, confusion_matrix):
    DCF = computeDCF(pi, Cfn, Cfp, confusion_matrix)
    dummy_function = np.array([pi * Cfn, (1 - pi) * Cfp])
    index = np.argmin(dummy_function)
    min_value = dummy_function[index]
    return DCF / min_value


def minDCFFromScores(pi, Cfn, Cfp, log_likelihood_ratios, labels):
    DCF_values = []

    for thresh in np.sort(log_likelihood_ratios):
        predictions = (log_likelihood_ratios > thresh).astype(int)

        confusion_matrix = confusionMatrix(predictions, labels, 2)
        DCF_values.append(computeNormalizedDCF(pi, Cfn, Cfp, confusion_matrix))

    index = np.argmin(DCF_values)
    min_DCF = DCF_values[index]
    return min_DCF

def minDCF(classifier, dataset, labels, pi, Cfn, Cfp, KNN=False):
    if KNN:
        return knnDCF(classifier, dataset, labels, pi, Cfn, Cfp)
    
    scores = classifier.getScores(dataset)
    DCF_values = []

    for thresh in np.sort(scores):
        predictions = (scores > thresh).astype(int)
        confusion_matrix = confusionMatrix(predictions, labels, 2)
        DCF_values.append(computeNormalizedDCF(pi, Cfn, Cfp, confusion_matrix))

    index = np.argmin(DCF_values)
    min_DCF = DCF_values[index]
    return min_DCF

def knnDCF(classifier, dataset, labels, pi, Cfn, Cfp):
    predictions = classifier.predict(dataset).astype(int)
    confusion_matrix = confusionMatrix(predictions, labels, 2)
    return computeNormalizedDCF(pi, Cfn, Cfp, confusion_matrix)

def computeOptimalThreshold(y_true, y_pred, cost_fn, cost_fp):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    min_cost = float('inf')
    optimal_threshold = thresholds[0]
    for threshold in thresholds:
        fnr = 1 - tpr[thresholds == threshold][0]
        fpr_ = fpr[thresholds == threshold][0]
        cost = cost_fn * fnr + cost_fp * fpr_
        if cost < min_cost:
            min_cost = cost
            optimal_threshold = threshold
    return optimal_threshold, min_cost

def actDCFFromScores(pi, Cfn, Cfp, log_likelihood_ratios, labels):
    predictions = (log_likelihood_ratios > (-np.log(pi/(1-pi)))).astype(int)
    confusion_matrix = confusionMatrix(predictions, labels, 2)
    act_DCF = computeNormalizedDCF(pi, Cfn, Cfp, confusion_matrix)
    return act_DCF

def actDCF(classifier, dataset, labels, pi, Cfn, Cfp):
    scores = classifier.getScores(dataset)
    predictions = (scores > (-np.log(pi/(1-pi)))).astype(int)
    confusion_matrix = confusionMatrix(predictions, labels, 2)
    act_DCF = computeNormalizedDCF(pi, Cfn, Cfp, confusion_matrix)
    return act_DCF


def calibrateScores(scores, sorted_labels, lambda_=1e-4, pi=0.5, SVM=False):
    pass
    scores = mrow(scores)
    logistic_regression = LogisticRegression()
    logistic_regression.train(scores, sorted_labels, lambda_, pi, SVM)
    alpha = logistic_regression.x_estimated_minimum_position[0]
    beta = logistic_regression.x_estimated_minimum_position[1]
    calibrated_scores = alpha * scores + beta - np.log(pi / (1 - pi))
    return calibrated_scores


def getFolds(dataset, labels, K):
    np.random.seed(46)
    folds = []
    new_labels = []
    n = int(dataset.shape[1] / K)

    idx = np.random.permutation(dataset.shape[1])
    for i in range(K):
        folds.append(dataset[:, idx[(i * n) : ((i + 1) * n)]])
        new_labels.append(labels[idx[(i * n) : ((i + 1) * n)]])

    return folds, new_labels


def calibrate(classifier, dataset, labels, pi, Cfn, Cfp, K=4, actualDCF=False, lambda_calibration=1e-4, calibration=True, SVM=False):
    folds, new_labels = getFolds(dataset, labels, K)

    sorted_labels = []
    scores = []
    for i in range(K):
        train_dataset = []
        train_labels = []

        for j in range(K):
            if j != i:
                train_dataset.append(folds[j])
                train_labels.append(new_labels[j])

        validation_dataset = folds[i]
        sorted_labels.append(new_labels[i])
        train_dataset = np.hstack(train_dataset)
        train_labels = np.hstack(train_labels)
        if SVM:
            if SVM[0] == 'Linear': # Linear SVM
                classifier.train(train_dataset, train_labels, SVM[1], SVM[2])
            elif SVM[0] == 'Polynomial': # Polynomial SVM
                classifier.train(train_dataset, train_labels, SVM[1], SVM[2], SVM[3], SVM[4])
            elif SVM[0] == 'RBF': # RBF SVM
                classifier.train(train_dataset, train_labels, SVM[1], SVM[2], SVM[3])
        else:
            classifier.train(train_dataset, train_labels)
        scores.append(classifier.getScores(validation_dataset))

    scores = np.hstack(scores)
    sorted_labels = np.hstack(sorted_labels)
    if calibration:
        scores = calibrateScores(scores, sorted_labels, lambda_calibration, pi).flatten() 
    new_labels = np.hstack(new_labels)

    if actualDCF:
        return actDCFFromScores(pi, Cfn, Cfp, scores, sorted_labels)
    return minDCFFromScores(pi, Cfn, Cfp, scores, sorted_labels)
