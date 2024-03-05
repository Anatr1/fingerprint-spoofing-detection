# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file contains the Gaussian Mixture Model classifiers

import numpy as np
import scipy
import utils

def logpdf_GAU_ND_1Sample(x, mu, C):
    const = - 0.5 * x.shape[0] * np.log(2 * np.pi)
    log_d = np.linalg.slogdet(C)[1]
    log_n = const - 0.5 * log_d - 0.5 * np.dot((x - mu).T,np.dot(np.linalg.inv(C), x - mu)).ravel()

    return log_n

def logpdf_GAU_ND(X,mu,C):         
    log_n = []
    for i in range(X.shape[1]):
        log_n.append(logpdf_GAU_ND_1Sample(X[:,i:i+1], mu, C)) 
    return np.array(log_n).ravel()

def getMaximumLikelihoodEstimation(X):
    mean = X.mean(1).reshape((X.shape[0],1))
    covariance = (np.dot(((X - mean)), (X - mean).T)) / X.shape[1]
    
    return mean, covariance

def EM_FullCov(data, gmm, psi, delta_loss):
    while True:
        component_count = len(gmm)
        log_likelihood = np.empty((component_count, data.shape[1]))

        # Expectation step
        for idx in range(component_count):
            log_likelihood[idx, :] = logpdf_GAU_ND(data, gmm[idx][1], gmm[idx][2])
            log_likelihood[idx, :] += np.log(gmm[idx][0])
        marginal_log_likelihood = utils.mrow(scipy.special.logsumexp(log_likelihood, axis=0))
        posterior_log_likelihood = log_likelihood - marginal_log_likelihood
        posterior_likelihood = np.exp(posterior_log_likelihood)

        previous_loss = marginal_log_likelihood

        # Maximization step
        Z = np.sum(posterior_likelihood, axis=1)

        F = np.zeros((data.shape[0], component_count))
        for idx in range(component_count):
            F[:, idx] = np.sum(posterior_likelihood[idx, :] * data, axis=1)

        S = np.zeros((data.shape[0], data.shape[0], component_count))
        for idx in range(component_count):
            S[:, :, idx] = np.dot(posterior_likelihood[idx, :] * data, data.T)

        mu = F / Z
        covariance = S / Z

        for idx in range(component_count):
            covariance[:, :, idx] -= np.dot(utils.mcol(mu[:, idx]), utils.mrow(mu[:, idx]))
            U, singular_values, _ = np.linalg.svd(covariance[:, :, idx])
            singular_values[singular_values < psi] = psi
            covariance[:, :, idx] = np.dot(U, utils.mcol(singular_values) * U.T)
        weights = Z / np.sum(Z)

        gmm = [(weights[i], utils.mcol(mu[:, i]), covariance[:, :, i]) for i in range(component_count)]
        current_loss = utils.logpdf_GMM(data, gmm)

        if np.mean(current_loss) - np.mean(previous_loss) < delta_loss:
            break

    return gmm

def EM_DiagCov(data, gmm, psi, delta_loss):
    while True:
        component_count = len(gmm)
        log_likelihood = np.empty((component_count, data.shape[1]))

        # Expectation step
        for idx in range(component_count):
            log_likelihood[idx, :] = logpdf_GAU_ND(data, gmm[idx][1], gmm[idx][2])
            log_likelihood[idx, :] += np.log(gmm[idx][0])
        marginal_log_likelihood = utils.mrow(scipy.special.logsumexp(log_likelihood, axis=0))
        posterior_log_likelihood = log_likelihood - marginal_log_likelihood
        posterior_likelihood = np.exp(posterior_log_likelihood)

        previous_loss = marginal_log_likelihood

        # Maximization step
        Z = np.sum(posterior_likelihood, axis=1)

        F = np.zeros((data.shape[0], component_count))
        for idx in range(component_count):
            F[:, idx] = np.sum(posterior_likelihood[idx, :] * data, axis=1)

        S = np.zeros((data.shape[0], data.shape[0], component_count))
        for idx in range(component_count):
            S[:, :, idx] = np.dot(posterior_likelihood[idx, :] * data, data.T)

        mu = F / Z
        covariance = S / Z

        for idx in range(component_count):
            covariance[:, :, idx] -= np.dot(utils.mcol(mu[:, idx]), utils.mrow(mu[:, idx]))

        weights = Z / np.sum(Z)

        for i in range(component_count):
            covariance[:, :, i] = np.diag(np.diag(covariance[:, :, i]))
            U, singular_values, _ = np.linalg.svd(covariance[:, :, i])
            singular_values[singular_values < psi] = psi
            covariance[:, :, i] = np.dot(U, utils.mcol(singular_values) * U.T)

        gmm = [(weights[i], utils.mcol(mu[:, i]), covariance[:, :, i]) for i in range(component_count)]
        current_loss = utils.logpdf_GMM(data, gmm)

        if np.mean(current_loss) - np.mean(previous_loss) < delta_loss:
            break

    return gmm

def EM_TiedCov(data, gmm, psi, delta_loss):
    while True:
        component_count = len(gmm)
        log_likelihood = np.empty((component_count, data.shape[1]))

        # Expectation step
        for idx in range(component_count):
            log_likelihood[idx, :] = logpdf_GAU_ND(data, gmm[idx][1], gmm[idx][2])
            log_likelihood[idx, :] += np.log(gmm[idx][0])
        marginal_log_likelihood = utils.mrow(scipy.special.logsumexp(log_likelihood, axis=0))
        posterior_log_likelihood = log_likelihood - marginal_log_likelihood
        posterior_likelihood = np.exp(posterior_log_likelihood)

        previous_loss = marginal_log_likelihood

        # Maximization step
        Z = np.sum(posterior_likelihood, axis=1)

        F = np.zeros((data.shape[0], component_count))
        for idx in range(component_count):
            F[:, idx] = np.sum(posterior_likelihood[idx, :] * data, axis=1)

        S = np.zeros((data.shape[0], data.shape[0], component_count))
        for idx in range(component_count):
            S[:, :, idx] = np.dot(posterior_likelihood[idx, :] * data, data.T)

        mu = F / Z
        covariance = S / Z

        for idx in range(component_count):
            covariance[:, :, idx] -= np.dot(utils.mcol(mu[:, idx]), utils.mrow(mu[:, idx]))
        weights = Z / np.sum(Z)

        updated_covariance = np.zeros(covariance[:, :, 0].shape)
        for idx in range(component_count):
            updated_covariance += (1 / data.shape[1]) * (Z[idx] * covariance[:, :, idx])

        U, singular_values, _ = np.linalg.svd(updated_covariance)
        singular_values[singular_values < psi] = psi
        updated_covariance = np.dot(U, utils.mcol(singular_values) * U.T)
        gmm = [(weights[i], utils.mcol(mu[:, i]), updated_covariance) for i in range(component_count)]

        current_loss = utils.logpdf_GMM(data, gmm)

        if np.mean(current_loss) - np.mean(previous_loss) < delta_loss:
            break

    return gmm

def LBGFullCov(samples, num_components, psi, alpha, delta_loss):
    start_weight = 1
    start_mu, start_cov = getMaximumLikelihoodEstimation(samples)

    U, singular_values, _ = np.linalg.svd(start_cov)
    singular_values[singular_values < psi] = psi
    start_cov = np.dot(U, utils.mcol(singular_values) * U.T)

    gmm_start = []
    if num_components == 1:
        gmm_start.append((start_weight, utils.mcol(start_mu), start_cov))
        return gmm_start

    new_weight = start_weight / 2
    U, singular_values, _ = np.linalg.svd(start_cov)
    d = U[:, 0:1] * singular_values[0]**0.5 * alpha
    new_mu_1 = start_mu + d
    new_mu_2 = start_mu - d
    gmm_start.append((new_weight, utils.mcol(new_mu_1), start_cov))
    gmm_start.append((new_weight, utils.mcol(new_mu_2), start_cov))

    while True:
        gmm_start = EM_FullCov(samples, gmm_start, psi, delta_loss)
        component_count = len(gmm_start)

        if component_count == num_components:
            break

        new_gmm = []
        for idx in range(component_count):
            new_weight = gmm_start[idx][0] / 2
            U, singular_values, Vh = np.linalg.svd(gmm_start[idx][2])
            d = U[:, 0:1] * singular_values[0]**0.5 * alpha
            new_mu_1 = gmm_start[idx][1] + d
            new_mu_2 = gmm_start[idx][1] - d
            new_gmm.append((new_weight, utils.mcol(new_mu_1), gmm_start[idx][2]))
            new_gmm.append((new_weight, utils.mcol(new_mu_2), gmm_start[idx][2]))
        gmm_start = new_gmm

    return gmm_start

def LBGDiagCov(samples, num_components, psi, alpha, delta_loss):
    start_weight = 1
    start_mu, start_cov = getMaximumLikelihoodEstimation(samples)

    start_cov = np.diag(np.diag(start_cov))

    U, singular_values, _ = np.linalg.svd(start_cov)
    singular_values[singular_values < psi] = psi
    start_cov = np.dot(U, utils.mcol(singular_values) * U.T)

    gmm_start = []
    if num_components == 1:
        gmm_start.append((start_weight, utils.mcol(start_mu), start_cov))
        return gmm_start

    new_weight = start_weight / 2
    U, singular_values, _ = np.linalg.svd(start_cov)
    d = U[:, 0:1] * singular_values[0]**0.5 * alpha
    new_mu_1 = start_mu + d
    new_mu_2 = start_mu - d
    gmm_start.append((new_weight, utils.mcol(new_mu_1), start_cov))
    gmm_start.append((new_weight, utils.mcol(new_mu_2), start_cov))

    while True:
        gmm_start = EM_DiagCov(samples, gmm_start, psi, delta_loss)
        component_count = len(gmm_start)

        if component_count == num_components:
            break

        new_gmm = []
        for idx in range(component_count):
            new_weight = gmm_start[idx][0] / 2
            U, singular_values, Vh = np.linalg.svd(gmm_start[idx][2])
            d = U[:, 0:1] * singular_values[0]**0.5 * alpha
            new_mu_1 = gmm_start[idx][1] + d
            new_mu_2 = gmm_start[idx][1] - d
            new_gmm.append((new_weight, utils.mcol(new_mu_1), gmm_start[idx][2]))
            new_gmm.append((new_weight, utils.mcol(new_mu_2), gmm_start[idx][2]))
        gmm_start = new_gmm

    return gmm_start  

def LBGTiedCov(samples, num_components, psi, alpha, delta_loss):
    start_weight = 1
    start_mu, start_cov = getMaximumLikelihoodEstimation(samples)

    U, singular_values, _ = np.linalg.svd(start_cov)
    singular_values[singular_values < psi] = psi
    start_cov = np.dot(U, utils.mcol(singular_values) * U.T)

    gmm_start = []
    if num_components == 1:
        gmm_start.append((start_weight, utils.mcol(start_mu), start_cov))
        return gmm_start

    new_weight = start_weight / 2
    U, singular_values, _ = np.linalg.svd(start_cov)
    d = U[:, 0:1] * singular_values[0]**0.5 * alpha
    new_mu_1 = start_mu + d
    new_mu_2 = start_mu - d
    gmm_start.append((new_weight, utils.mcol(new_mu_1), start_cov))
    gmm_start.append((new_weight, utils.mcol(new_mu_2), start_cov))

    while True:
        gmm_start = EM_TiedCov(samples, gmm_start, psi, delta_loss)
        component_count = len(gmm_start)

        if component_count == num_components:
            break

        new_gmm = []
        for idx in range(component_count):
            new_weight = gmm_start[idx][0] / 2
            U, singular_values, Vh = np.linalg.svd(gmm_start[idx][2])
            d = U[:, 0:1] * singular_values[0]**0.5 * alpha
            new_mu_1 = gmm_start[idx][1] + d
            new_mu_2 = gmm_start[idx][1] - d
            new_gmm.append((new_weight, utils.mcol(new_mu_1), gmm_start[idx][2]))
            new_gmm.append((new_weight, utils.mcol(new_mu_2), gmm_start[idx][2]))
        gmm_start = new_gmm

    return gmm_start


class GMMFullCovariance:
    def __init__(self):
        self.id_string = "--------------------- GMM FULL COVARIANCE ----------------------"
        self.num_components = 1
        self.num_classes = 1
        self.GMMs = []

    def train(self, dataset_training, labels_training, num_components, num_classes=2, psi=1e-2, alpha=1e-1, loss_delta=1e-6):
        self.num_components = num_components
        self.num_classes = num_classes

        for i in range(num_classes):
            samples = dataset_training[:, labels_training == i]
            self.GMMs.append(LBGFullCov(samples, num_components, psi, alpha, loss_delta))

    def getScores(self, X):
        scores = np.zeros((self.num_classes, X.shape[1]))

        for i in range(self.num_classes):
            scores[i, :] = utils.logpdf_GMM(X, self.GMMs[i])

        scores = scores[1] - scores[0]
        return scores

    def predict(self, X):
        scores = np.zeros((self.num_classes, X.shape[1]))

        for i in range(self.num_classes):
            scores[i, :] = utils.logpdf_GMM(X, self.GMMs[i])

        return np.argmax(scores, axis=0)

class GMMNaiveBayes:
    def __init__(self):
        self.id_string = "--------------------- GMM NAIVE BAYES ----------------------"
        self.num_components = 1
        self.num_classes = 1
        self.GMMs = []      

    def train(self, dataset_training, labels_training, num_components, num_classes=2, psi=1e-2, alpha=1e-1, loss_delta=1e-6):
        self.num_components = num_components
        self.num_classes = num_classes

        for i in range(num_classes):
            samples = dataset_training[:, labels_training == i]
            self.GMMs.append(LBGDiagCov(samples, num_components, psi, alpha, loss_delta))

    def getScores(self, X):
        scores = np.zeros((self.num_classes, X.shape[1]))

        for i in range(self.num_classes):
            scores[i, :] = utils.logpdf_GMM(X, self.GMMs[i])

        scores = scores[1] - scores[0]
        return scores

    def predict(self, X):
        scores = np.zeros((self.num_classes, X.shape[1]))

        for i in range(self.num_classes):
            scores[i, :] = utils.logpdf_GMM(X, self.GMMs[i])

        return np.argmax(scores, axis=0)    

class GMMTiedCovariance:
    def __init__(self):
        self.id_string = "--------------------- GMM TIED COVARIANCE ----------------------"
        self.num_components = 1
        self.num_classes = 1
        self.GMMs = []
    
    def train(self, dataset_training, labels_training, num_components, num_classes=2, psi=1e-2, alpha=1e-1, loss_delta=1e-6):
        self.num_components = num_components
        self.num_classes = num_classes

        for i in range(num_classes):
            samples = dataset_training[:, labels_training == i]
            self.GMMs.append(LBGTiedCov(samples, num_components, psi, alpha, loss_delta))

    def getScores(self, X):
        scores = np.zeros((self.num_classes, X.shape[1]))

        for i in range(self.num_classes):
            scores[i, :] = utils.logpdf_GMM(X, self.GMMs[i])

        scores = scores[1] - scores[0]
        return scores

    def predict(self, X):
        scores = np.zeros((self.num_classes, X.shape[1]))

        for i in range(self.num_classes):
            scores[i, :] = utils.logpdf_GMM(X, self.GMMs[i])

        return np.argmax(scores, axis=0)
