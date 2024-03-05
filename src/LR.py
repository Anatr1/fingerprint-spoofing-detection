# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file contains the Logistic Regression classifier.

import numpy as np
import scipy.optimize
import utils

MAX_DIMENSIONS = 10

def J_binary(w, b, dataset, labels, lambda_, prior, SVM=False):
    norm_term = lambda_ / 2 * (np.linalg.norm(w) ** 2)
    sum_term_positive = 0
    sum_term_negative = 0

    for i in range(dataset.shape[1]):
        if labels[i] == 1:
            sum_term_positive += np.log1p(np.exp(-np.dot(w.T, dataset[:, i]) - b))
        else:
            sum_term_negative += np.log1p(np.exp(np.dot(w.T, dataset[:, i]) + b))

    if not SVM:
        j = (
                norm_term
                + (prior / dataset[:, labels == 1].shape[1]) * sum_term_positive
                + ((1 - prior) / dataset[:, labels == 0].shape[1]) * sum_term_negative
        )   
    else:
        j = (
                norm_term
                + (prior / dataset[:, labels == 1].shape[1]) * sum_term_positive
                + ((1 - prior) / dataset[:, labels == -1].shape[1]) * sum_term_negative
        )
    
    return j

def binary_logreg_obj(v, dataset, labels, lambda_, prior=0.5, SVM=False):
    w, b = v[0:-1], v[-1]

    j = J_binary(w, b, dataset, labels, lambda_, prior, SVM)
    return j

def J_multiclass(w, b, dataset, labels, lambda_):
    norm_term = lambda_ / 2 * (np.linalg.norm(w) ** 2)
    sum_term = 0

    for i in range(dataset.shape[1]):
        w_reshaped = w[:dataset.shape[0]].reshape(-1, 1)
        dataset_reshaped = dataset[:, i].reshape(-1, 1)
        sum_term += np.log1p(np.exp(-np.dot(w_reshaped.T, dataset_reshaped) - b[labels[i]]))

    j = norm_term + (1 / dataset.shape[1]) * sum_term

    return j

def vecxxT(x):
    x = x[:, None]
    xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
    return xxT

def multiclass_logreg_obj(v, dataset, labels, lambda_):
    w, b = v[0:-len(set(labels))], v[-len(set(labels)):]

    j = J_multiclass(w, b, dataset, labels, lambda_)
    return j

def compute_gradient(parameters, regularization_param, data, labels, prior, num_true, num_false):        
    weights, bias = parameters[:-1], parameters[-1]

    z = 2 * labels - 1

    first_term = regularization_param * weights
    second_term = 0        
    third_term = 0

    for i in range(data.shape[1]):
        S = np.dot(weights.T, data[:, i]) + bias            
        z_S = z[i] * S
        if labels[i] == 1:
            internal_term = np.exp(-z_S) * (-z[i] * data[:, i]) / (1 + np.exp(-z_S))                
            second_term += internal_term            
        else:
            internal_term_2 = np.exp(-z_S) * (-z[i] * data[:, i]) / (1 + np.exp(-z_S))                
            third_term += internal_term_2

    derivative_weights = first_term + (prior / num_true) * second_term + (1 - prior) / num_false * third_term

    first_term = 0                   
    second_term = 0
    for i in range(data.shape[1]):
        S = np.dot(weights.T, data[:, i]) + bias
        z_S = z[i] * S
        if labels[i] == 1:                
            internal_term = np.exp(-z_S) * (-z[i]) / (1 + np.exp(-z_S))
            first_term += internal_term            
        else:
            internal_term_2 = np.exp(-z_S) * (-z[i]) / (1 + np.exp(-z_S))                
            second_term += internal_term_2
       
    derivative_bias = (prior / num_true) * first_term + (1 - prior) / num_false * second_term        
    gradient = np.hstack((derivative_weights, derivative_bias))
    return gradient

def quadratic_logreg_obj(v, phi, labels, lambda_, prior=0.5):
    loss_c0 = 0
    loss_c1 = 0

    w, b = v[0:-1], v[-1]
    w = utils.mcol(w)
    n = phi.shape[1]

    n_true = len(np.where(labels == 1)[0])
    n_false = len(np.where(labels == 0)[0])
    
    regularization = (lambda_ / 2) * np.sum(w ** 2) 
    
    for i in range(n):
        if (labels[i:i+1] == 1):
            zi = 1
            loss_c1 += np.logaddexp(0, -zi * (np.dot(w.T, phi[:,i:i+1]) + b))
        else:
            zi=-1
            loss_c0 += np.logaddexp(0, -zi * (np.dot(w.T, phi[:,i:i+1]) + b))
    
    J = regularization + (prior / n_true) * loss_c1 + (1 - prior) / n_false * loss_c0

    gradient = compute_gradient(v, lambda_, phi, labels, prior, n_true, n_false)

    return J, gradient


class LogisticRegression:
    def __init__(self) -> None:
        self.id_string = "--------------------- LOGISTIC REGRESSION ----------------------"
        self.x_estimated_minimum_position = 0
        self.f_objective_value_at_minimum = 0
        self.d_info = ""

    def train(self, dataset_training, labels_training, lambda_, prior=0.5, SVM=False):
        (
            self.x_estimated_minimum_position,
            self.f_objective_value_at_minimum,
            self.d_info,
        ) = scipy.optimize.fmin_l_bfgs_b(
            binary_logreg_obj,
            np.zeros(dataset_training.shape[0] + 1),
            args=(dataset_training, labels_training, lambda_, prior, SVM),
            approx_grad=True,
        )

    def getScores(self, X):
        scores = np.dot(self.x_estimated_minimum_position[0:-1], X) + self.x_estimated_minimum_position[-1]
        return scores

    def predict(self, X):
        scores = self.getScores(X)
        prediction = (scores > 0).astype(int)
        return prediction
    
class MulticlassLogisticRegression:
    def __init__(self) -> None:
        self.id_string = "--------------------- MULTICLASS LOGISTIC REGRESSION ----------------------"
        self.x_estimated_minimum_position = 0
        self.f_objective_value_at_minimum = 0
        self.d_info = ""

    def train(self, dataset_training, labels_training, lambda_):
        (
            self.x_estimated_minimum_position,
            self.f_objective_value_at_minimum,
            self.d_info,
        ) = scipy.optimize.fmin_l_bfgs_b(
            multiclass_logreg_obj,
            np.zeros((dataset_training.shape[0], MAX_DIMENSIONS + 1)),
            args=(dataset_training, labels_training, lambda_),
            approx_grad=True,
        )

    def getScores(self, X):
        # Ensure that x_estimated_minimum_position and X have compatible shapes for the dot product operation
        x_estimated_minimum_position_reshaped = self.x_estimated_minimum_position[:X.shape[0]].reshape(-1, 1)
        scores = np.dot(x_estimated_minimum_position_reshaped.T, X) + self.x_estimated_minimum_position[-1]
        return scores

    def predict(self, X):
        scores = self.getScores(X)
        prediction = np.argmax(scores, axis=0)
        return prediction

class QuadraticLogisticRegression:
    def __init__(self) -> None:
        self.id_string = "--------------------- QUADRATIC LOGISTIC REGRESSION ----------------------"
        self.x_estimated_minimum_position = 0
        self.f_objective_value_at_minimum = 0
        self.d_info = ""
        self.phi = 0
        self.w = 0
        self.b = 0

    def train(self, dataset_training, labels_training, lambda_, prior=0.5):
        expanded_dataset = np.apply_along_axis(vecxxT, 0, dataset_training)
        self.phi = np.vstack((dataset_training, expanded_dataset))

        (
            self.x_estimated_minimum_position,
            self.f_objective_value_at_minimum,
            self.d_info,
        ) = scipy.optimize.fmin_l_bfgs_b(
            quadratic_logreg_obj,
            np.zeros(self.phi.shape[0] + 1),
            args=(self.phi, labels_training, lambda_, prior),
            #approx_grad=True,
        )

        self.w = np.array(self.x_estimated_minimum_position[0:-1])
        self.b = self.x_estimated_minimum_position[-1]

    def getScores(self, X):
        scores = []
       
        for i in range(X.shape[1]):
            x = utils.mcol(X[:,i:i+1])
            vec_x = utils.mcol(np.hstack(np.dot(x, x.T)))
            phi = np.vstack((vec_x, x))
            scores.append(np.dot(self.w.T, phi) + self.x_estimated_minimum_position[-1])
        
        return scores

    def predict(self, X):
        scores = np.array(self.getScores(X))
        prediction = (scores > 0).astype(int)
        prediction = prediction.flatten()
        return prediction
