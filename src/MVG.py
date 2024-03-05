# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file contains the Multivariate Gaussian classifier.


import utils
import scipy
import numpy as np

class GaussianClassifierFullCovariance:
    def __init__(self) -> None:
        self.id_string = "--------------------- GAUSSIAN CLASSIFIER FULL COVARIANCE ----------------------"
        self.mean_class_0 = 0
        self.mean_class_1 = 0
        self.sigma_class_0 = 0
        self.sigma_class_1 = 0
        self.pi_class_0 = 0
        self.pi_class_1 = 0

    def train(self, dataset, labels):
        self.mean_class_0 = utils.mcol(dataset[:, labels == 0].mean(axis=1))
        self.mean_class_1 = utils.mcol(dataset[:, labels == 1].mean(axis=1))

        self.sigma_class_0 = np.cov(dataset[:, labels == 0])
        self.sigma_class_1 = np.cov(dataset[:, labels == 1])

        self.pi_class_0 = dataset[:, labels == 0].shape[1] / dataset.shape[1]
        self.pi_class_1 = dataset[:, labels == 1].shape[1] / dataset.shape[1]

    def getScores(self, X):
        ll_class_0 = utils.getLogLikelihood(X, self.mean_class_0, self.sigma_class_0)
        ll_class_1 = utils.getLogLikelihood(X, self.mean_class_1, self.sigma_class_1)

        scores = ll_class_1 - ll_class_0
        return scores

    def predict(self, X):
        ll_class_0 = utils.getLogLikelihood(X, self.mean_class_0, self.sigma_class_0)
        ll_class_1 = utils.getLogLikelihood(X, self.mean_class_1, self.sigma_class_1)

        S = np.vstack((ll_class_0, ll_class_1))

        S_joint = S + utils.mcol(
            np.array(
                [
                    np.log(self.pi_class_0),
                    np.log(self.pi_class_1),
                ]
            )
        )

        # Compute marginal log densities
        marginal_log_densities = utils.mrow(scipy.special.logsumexp(S_joint, axis=0))

        # Get predictions
        log_posteriors = S_joint - marginal_log_densities
        predictions = np.argmax(log_posteriors, axis=0)

        return predictions

    def logInfo(self):
        print(f"\nGaussianClassifierFullCovariance:")
        print(f"\nMean class 0:\n{self.mean_class_0}")
        print(f"\nMean class 1:\n{self.mean_class_1}")
        print(f"\nSigma class 0:\n{self.sigma_class_0}")
        print(f"\nSigma class 1:\n{self.sigma_class_1}")
        print(f"\nPi class 0:\n{self.pi_class_0}")
        print(f"\nPi class 1:\n{self.pi_class_1}")

class GaussianClassifierNaiveBayes:
    def __init__(self) -> None:
        self.id_string = "--------------------- GAUSSIAN CLASSIFIER NAIVE BAYES ----------------------"
        self.mean_class_0 = 0
        self.mean_class_1 = 0
        self.sigma_class_0 = 0
        self.sigma_class_1 = 0
        self.pi_class_0 = 0
        self.pi_class_1 = 0

    def train(self, dataset, labels):
        self.mean_class_0 = utils.mcol(dataset[:, labels == 0].mean(axis=1))
        self.mean_class_1 = utils.mcol(dataset[:, labels == 1].mean(axis=1))

        identity_matrix = np.identity(dataset.shape[0])
        self.sigma_class_0 = np.multiply(
            np.cov(dataset[:, labels == 0]), identity_matrix
        )
        self.sigma_class_1 = np.multiply(
            np.cov(dataset[:, labels == 1]), identity_matrix
        )

        self.pi_class_0 = dataset[:, labels == 0].shape[1] / dataset.shape[1]
        self.pi_class_1 = dataset[:, labels == 1].shape[1] / dataset.shape[1]

    def getScores(self, X):
        ll_class_0 = utils.getLogLikelihood(X, self.mean_class_0, self.sigma_class_0)
        ll_class_1 = utils.getLogLikelihood(X, self.mean_class_1, self.sigma_class_1)

        scores = ll_class_1 - ll_class_0
        return scores

    def predict(self, X):
        ll_class_0 = utils.getLogLikelihood(X, self.mean_class_0, self.sigma_class_0)
        ll_class_1 = utils.getLogLikelihood(X, self.mean_class_1, self.sigma_class_1)

        S = np.vstack((ll_class_0, ll_class_1))

        S_joint = S + utils.mcol(
            np.array(
                [
                    np.log(self.pi_class_0),
                    np.log(self.pi_class_1),
                ]
            )
        )

        # Compute marginal log densities
        marginal_log_densities = utils.mrow(scipy.special.logsumexp(S_joint, axis=0))

        # Get predictions
        log_posteriors = S_joint - marginal_log_densities
        predictions = np.argmax(log_posteriors, axis=0)

        return predictions

    def logInfo(self):
        print(f"\nGaussianClassifierNaiveBayes:")
        print(f"\nMean class 0:\n{self.mean_class_0}")
        print(f"\nMean class 1:\n{self.mean_class_1}")
        print(f"\nSigma class 0:\n{self.sigma_class_0}")
        print(f"\nSigma class 1:\n{self.sigma_class_1}")
        print(f"\nPi class 0:\n{self.pi_class_0}")
        print(f"\nPi class 1:\n{self.pi_class_1}")

class GaussianClassifierFullTiedCovariance:
    def __init__(self) -> None:
        self.id_string = "--------------------- GAUSSIAN CLASSIFIER FULL TIED COVARIANCE ----------------------"
        self.mean_class_0 = 0
        self.mean_class_1 = 0
        self.sigma_class_0 = 0
        self.sigma_class_1 = 0
        self.sigma_tied = 0
        self.pi_class_0 = 0
        self.pi_class_1 = 0

    def train(self, dataset, labels):
        self.mean_class_0 = utils.mcol(dataset[:, labels == 0].mean(axis=1))
        self.mean_class_1 = utils.mcol(dataset[:, labels == 1].mean(axis=1))

        self.sigma_class_0 = np.cov(dataset[:, labels == 0])
        self.sigma_class_1 = np.cov(dataset[:, labels == 1])
        
        self.sigma_tied = (
            1
            / (dataset.shape[1])
            * (
                dataset[:, labels == 0].shape[1] * self.sigma_class_0
                + dataset[:, labels == 1].shape[1] * self.sigma_class_1
            )
        )

        self.pi_class_0 = dataset[:, labels == 0].shape[1] / dataset.shape[1]
        self.pi_class_1 = dataset[:, labels == 1].shape[1] / dataset.shape[1]

    def getScores(self, X):
        ll_class_0 = utils.getLogLikelihood(X, self.mean_class_0, self.sigma_tied)
        ll_class_1 = utils.getLogLikelihood(X, self.mean_class_1, self.sigma_tied)

        scores = ll_class_1 - ll_class_0
        return scores

    def predict(self, X):
        ll_class_0 = utils.getLogLikelihood(X, self.mean_class_0, self.sigma_tied)
        ll_class_1 = utils.getLogLikelihood(X, self.mean_class_1, self.sigma_tied)
        
        S = np.vstack((ll_class_0, ll_class_1))

        S_joint = S + utils.mcol(
            np.array(
                [
                    np.log(self.pi_class_0),
                    np.log(self.pi_class_1),
                ]
            )
        )
        
        # Compute marginal log densities
        marginal_log_densities = utils.mrow(scipy.special.logsumexp(S_joint, axis=0))

        # Get predictions
        log_posteriors = S_joint - marginal_log_densities
        predictions = np.argmax(log_posteriors, axis=0)

        return predictions

    def logInfo(self):
        print(f"\nGaussianClassifierFull:")
        print(f"\nMean class 0:\n{self.mean_class_0}")
        print(f"\nMean class 1:\n{self.mean_class_1}")
        print(f"\nSigma class 0:\n{self.sigma_class_0}")
        print(f"\nSigma class 1:\n{self.sigma_class_1}")
        print(f"\nSigma tied:\n{self.sigma_tied}")
        print(f"\nPi class 0:\n{self.pi_class_0}")
        print(f"\nPi class 1:\n{self.pi_class_1}")

class GaussianClassifierNaiveBayesTiedCovariance:
    def __init__(self) -> None:
        self.id_string = "--------------------- GAUSSIAN CLASSIFIER NAIVE BAYES TIED COVARIANCE ----------------------"
        self.mean_class_0 = 0
        self.mean_class_1 = 0
        self.sigma_class_0 = 0
        self.sigma_class_1 = 0
        self.sigma_tied = 0
        self.pi_class_0 = 0
        self.pi_class_1 = 0

    def train(self, dataset, labels):
        self.mean_class_0 = utils.mcol(dataset[:, labels == 0].mean(axis=1))
        self.mean_class_1 = utils.mcol(dataset[:, labels == 1].mean(axis=1))

        identity_matrix = np.identity(dataset.shape[0])
        self.sigma_class_0 = np.multiply(
            np.cov(dataset[:, labels == 0]), identity_matrix
        )
        self.sigma_class_1 = np.multiply(
            np.cov(dataset[:, labels == 1]), identity_matrix
        )

        self.sigma_tied = (
            1 / (dataset.shape[1]) * (
                dataset[:, labels == 0].shape[1] * self.sigma_class_0
                + dataset[:, labels == 1].shape[1] * self.sigma_class_1
            )
        )

        self.pi_class_0 = dataset[:, labels == 0].shape[1] / dataset.shape[1]
        self.pi_class_1 = dataset[:, labels == 1].shape[1] / dataset.shape[1]

    def getScores(self, X):
        ll_class_0 = utils.getLogLikelihood(X, self.mean_class_0, self.sigma_tied)
        ll_class_1 = utils.getLogLikelihood(X, self.mean_class_1, self.sigma_tied)

        scores = ll_class_1 - ll_class_0
        return scores

    def predict(self, X):
        ll_class_0 = utils.getLogLikelihood(X, self.mean_class_0, self.sigma_tied)
        ll_class_1 = utils.getLogLikelihood(X, self.mean_class_1, self.sigma_tied)

        S = np.vstack((ll_class_0, ll_class_1))

        S_joint = S + utils.mcol(
            np.array(
                [
                    np.log(self.pi_class_0),
                    np.log(self.pi_class_1),
                ]
            )
        )

        # Compute marginal log densities
        marginal_log_densities = utils.mrow(scipy.special.logsumexp(S_joint, axis=0))

        # Get predictions
        log_posteriors = S_joint - marginal_log_densities
        predictions = np.argmax(log_posteriors, axis=0)

        return predictions
    
    def logInfo(self):
        print(f"\nGaussianClassifierNaiveBayesTiedCovariance:")
        print(f"\nMean class 0:\n{self.mean_class_0}")
        print(f"\nMean class 1:\n{self.mean_class_1}")
        print(f"\nSigma class 0:\n{self.sigma_class_0}")
        print(f"\nSigma class 1:\n{self.sigma_class_1}")
        print(f"\nSigma tied:\n{self.sigma_tied}")
        print(f"\nPi class 0:\n{self.pi_class_0}")
        print(f"\nPi class 1:\n{self.pi_class_1}")
