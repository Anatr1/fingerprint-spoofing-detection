# Fingerprint Spoofing Detection

![Fingerprint Spoofing Detection](./img/fingerprints.png)

_[Project developed for the Machine Learning and Pattern Recognition course @Politecnico di Torino]_

This project aims to evaluate the performance of various machine learning supervised algorithms in distinguishing between genuine and spoofed fingerprint images. Specifically, it focuses on detecting fingerprints that have been maliciously replicated, posing a potential security threat.

## Dataset
The project uses a dataset of synthetic fingerprint images, represented by 10-dimensional embeddings. The dataset has two classes: authentic and spoofed fingerprints. The spoofed class has six sub-classes corresponding to different spoofing methods, but the sub-class labels are not available. The dataset is pre-divided into a training set (2,325 samples) and a test set (7,704 samples), with a class imbalance ratio of about 2:16.

## Classifiers
The project employs and compares the following classification techniques:

- **K-Nearest Neighbor (KNN) Classifier**: a simple, non-parametric and computationally efficient method that assigns new fingerprints to their appropriate class based on their nearest neighbors in the training set.
- **Multivariate Gaussian (MVG) Classifiers**: generative models that assume that each class follows a multivariate normal distribution, with different covariance structures (full, diagonal, full tied, diagonal tied).
- **Gaussian Mixture Models (GMM)**: generative models that assume that each class is a mixture of several multivariate normal components, with different covariance structures (full, diagonal, full tied, diagonal tied).
- **Logistic Regression (LR)**: discriminative models that use linear or quadratic functions to separate the two classes, with a regularization term to control the model complexity.
- **Support Vector Machines (SVM)**: discriminative models that seek the optimal hyperplane that maximizes the margin between the two classes, with different kernel functions (linear, polynomial, radial basis function).

The project uses the minimum Detection Cost Function (minDCF) as the primary performance metric, which incorporates the relative costs associated with False Negative (FN) and False Positive (FP) errors. Additionally, error rate is used as a secondary metric. The project also performs score calibration to improve the model scores and reduce the gap between minDCF and actual DCF.

