# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file contains the functions to perform Leave One Out and K-Fold Cross Validation.

import numpy as np
import utils

def LOOCrossValidation(classifier, dataset, labels, lambda_=0, components=0, num_classes=2, SVM=False, KNN=False, DIMENSIONS=10, prior=0.5, Cfn=1, Cfp=1, actualDCF=False):
    # Performs Leave One Out Cross Validation on a given dataset and labels.
    # Returns the average error rate and the standard deviation.
    k = len(dataset[0])

    # Seed the random number generator for reproducibility
    np.random.seed(0)
    
    # Shuffle the dataset and labels
    dataset = np.array(dataset)
    labels = np.array(labels)
    indices = np.arange(len(dataset[0]))
    np.random.shuffle(indices)
    dataset = dataset[:, indices]
    labels = labels[indices]

    error_rate = []
    min_DCF = []

    print(f"Performing Leave One Out Cross Validation with {k} samples...")
    for i in range(k):
        #print(f"{i + 1}/{k}")
        training_dataset = []
        for dim in range(DIMENSIONS):
            training_dataset.append([])
        training_labels = []

        validation_dataset = []
        for dim in range(DIMENSIONS):
            validation_dataset.append([dataset[dim][i]])
        validation_labels = labels[i]
        
        for dim in range(DIMENSIONS):
            for j in range(k):
                if j != i:
                    training_dataset[dim].append(dataset[dim][j])
                    if len(training_labels) < k - 1:
                        training_labels.append(labels[j])

        training_dataset = np.array(training_dataset)
        training_labels = np.array(training_labels)
        validation_dataset = np.array(validation_dataset)
        validation_labels = np.array(validation_labels)

        if SVM is False:
            if components != 0: # GMM
                classifier.train(training_dataset, training_labels, components, num_classes)
            elif lambda_ != 0: # LR
                classifier.train(training_dataset, training_labels, lambda_)
            elif KNN is not False: # KNN
                classifier.fit(training_dataset, training_labels)
            else: # MVG
                classifier.train(training_dataset, training_labels)
        else:
            if SVM[0] == 'Linear': # Linear SVM
                classifier.train(training_dataset, training_labels, SVM[1], SVM[2])
            elif SVM[0] == 'Polynomial': # Polynomial SVM
                classifier.train(training_dataset, training_labels, SVM[1], SVM[2], SVM[3], SVM[4])
            elif SVM[0] == 'RBF': # RBF SVM
                classifier.train(training_dataset, training_labels, SVM[1], SVM[2], SVM[3])

        error_rate.append(utils.computeErrorRate(classifier.predict(validation_dataset), validation_labels))
        if actualDCF is False:
            min_DCF.append(utils.minDCF(classifier, validation_dataset, validation_labels, prior, Cfn, Cfp, KNN=KNN))
        else:
            min_DCF.append(utils.actDCF(classifier, validation_dataset, validation_labels, prior, Cfn, Cfp))

    error_rate = np.array(error_rate)
    print(f"Average error rate: {round(np.mean(error_rate), 2)}%")
    print(f"Standard deviation: {round(np.std(error_rate), 2)}%")

    min_DCF = np.array(min_DCF)
    print(f"Average minDCF: {round(np.mean(min_DCF), 3)}")
    print(f"Standard deviation: {round(np.std(min_DCF), 2)}")

    return np.mean(error_rate), np.std(error_rate), np.mean(min_DCF), np.std(min_DCF)

def KFoldCrossValidation(classifier, dataset, labels, k=3, lambda_= 0, components=0, num_classes=2, SVM=False, KNN=False, DIMENSIONS=10, prior=0.5, Cfn=1, Cfp=1, actualDCF=False):
    # Performs K-Fold Cross Validation on a given dataset and labels.
    # Returns the average error rate and the standard deviation.

    # Seed the random number generator for reproducibility
    np.random.seed(0)
    
    # Shuffle the dataset and labels
    dataset = np.array(dataset)
    labels = np.array(labels)
    indices = np.arange(len(dataset[0]))
    np.random.shuffle(indices)
    dataset = dataset[:, indices]
    labels = labels[indices]

    error_rate = []
    min_DCF = []

    print(f"Performing K-Fold Cross Validation with {k} folds...")
    for i in range(k):
        #print(f"{i + 1}/{k}")
        training_dataset = []
        for dim in range(DIMENSIONS):
            training_dataset.append([])
        training_labels = []

        validation_dataset = []
        for dim in range(DIMENSIONS):
            validation_dataset.append([])
        validation_labels = []

        for j in range(len(dataset[0])):
            if j % k == i:
                for dim in range(DIMENSIONS):
                    validation_dataset[dim].append(dataset[dim][j])
                validation_labels.append(labels[j])
            else:
                for dim in range(DIMENSIONS):
                    training_dataset[dim].append(dataset[dim][j])
                training_labels.append(labels[j])

        training_dataset = np.array(training_dataset)
        training_labels = np.array(training_labels)
        validation_dataset = np.array(validation_dataset)
        validation_labels = np.array(validation_labels)

        if SVM is False:
            if components != 0: # GMM
                classifier.train(training_dataset, training_labels, components, num_classes)
            elif lambda_ != 0: # LR
                classifier.train(training_dataset, training_labels, lambda_)
            elif KNN is not False: # KNN
                classifier.fit(training_dataset, training_labels)
            else: # MVG
                classifier.train(training_dataset, training_labels)
        else:
            if SVM[0] == 'Linear': # Linear SVM
                classifier.train(training_dataset, training_labels, SVM[1], SVM[2])
            elif SVM[0] == 'Polynomial': # Polynomial SVM
                classifier.train(training_dataset, training_labels, SVM[1], SVM[2], SVM[3], SVM[4])
            elif SVM[0] == 'RBF': # RBF SVM
                classifier.train(training_dataset, training_labels, SVM[1], SVM[2], SVM[3])
                
        error_rate.append(utils.computeErrorRate(classifier.predict(validation_dataset), validation_labels))
        if actualDCF is False:
            min_DCF.append(utils.minDCF(classifier, validation_dataset, validation_labels, prior, Cfn, Cfp, KNN=KNN))
        else:
            min_DCF.append(utils.actDCF(classifier, validation_dataset, validation_labels, prior, Cfn, Cfp))
        
    error_rate = np.array(error_rate)
    print(f"Average error rate: {round(np.mean(error_rate), 2)}%")
    print(f"Standard deviation: {round(np.std(error_rate), 2)}%")

    min_DCF = np.array(min_DCF)
    print(f"Average minDCF: {round(np.mean(min_DCF), 3)}")
    print(f"Standard deviation: {round(np.std(min_DCF), 2)}")

    return np.mean(error_rate), np.std(error_rate), np.mean(min_DCF), np.std(min_DCF)
