# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file contains the Nearest Neighbors Classifiers.

import numpy as np

class NearestNeighborClassifier:
    def __init__(self):
        self.id_string = "------------------ NEAREST NEIGHBORS ------------------"

    def fit(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def predict(self, dataset):
        predictions = np.zeros(dataset.shape[1])

        for i in range(dataset.shape[1]):
            distances = np.linalg.norm(self.dataset - dataset[:, i].reshape(-1, 1), axis=0)
            predictions[i] = self.labels[np.argmin(distances)]

        return predictions
    
class KNearestNeighborClassifier:
    def __init__(self, k=3):
        self.id_string = "------------------ K-NEAREST NEIGHBORS ------------------"
        self.k = k

    def fit(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def predict(self, dataset):
        predictions = np.zeros(dataset.shape[1])

        for i in range(dataset.shape[1]):
            distances = np.linalg.norm(self.dataset - dataset[:, i].reshape(-1, 1), axis=0)
            closest = np.argsort(distances)[:self.k]
            predictions[i] = np.argmax(np.bincount(self.labels[closest]))

        return predictions
    
class WeightedKNearestNeighborClassifier:
    def __init__(self, k=3):
        self.id_string = "------------------ WEIGHTED K-NEAREST NEIGHBORS ------------------"
        self.k = k

    def fit(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def predict(self, dataset):
        predictions = np.zeros(dataset.shape[1])

        for i in range(dataset.shape[1]):
            distances = np.linalg.norm(self.dataset - dataset[:, i].reshape(-1, 1), axis=0)
            closest = np.argsort(distances)[:self.k]
            predictions[i] = np.argmax(np.bincount(self.labels[closest], weights=1/distances[closest]))

        return predictions