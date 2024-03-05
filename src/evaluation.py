import utils
from PCA import PCA
from LDA import LDA
from SVM import RadialBasisFunctionSVM
from GMM import GMMFullCovariance
from NNC import KNearestNeighborClassifier, WeightedKNearestNeighborClassifier
from MVG import GaussianClassifierFullCovariance
from LR import LogisticRegression
import numpy as np

prior = 0.5
Cfn = 1
Cfp = 10

dataset, labels = utils.loadData("../data/Train.txt")
evaluation_dataset, evaluation_labels = utils.loadData("../data/Test.txt")
dataset_svm, labels_svm = utils.loadDataChangeLabels("../data/Train.txt")
evaluation_dataset_svm, evaluation_labels_svm = utils.loadDataChangeLabels("../data/Test.txt")

print(f"Evaluation for the application: Prior = {prior}, Cfn = {Cfn}, Cfp = {Cfp}")

############### K-Nearest Neighbors ###############
KNN = KNearestNeighborClassifier()

KNN.k = 5
KNN.fit(dataset, labels)
print(f"KNN Error Rate: {utils.computeErrorRate(KNN.predict(evaluation_dataset), evaluation_labels)}")
print(f"KNN MinDCF: {utils.minDCF(KNN, evaluation_dataset, evaluation_labels, prior, Cfn, Cfp, KNN=True)}\n")

############### Weighted K-Nearest Neighbors ###############
WKNN = WeightedKNearestNeighborClassifier()

WKNN.k = 7
WKNN.fit(dataset, labels)
print(f"WKNN Error Rate: {utils.computeErrorRate(WKNN.predict(evaluation_dataset), evaluation_labels)}")
print(f"WKNN MinDCF: {utils.minDCF(WKNN, evaluation_dataset, evaluation_labels, prior, Cfn, Cfp, KNN=True)}\n")

############### Logistic Regression ###############
lambda_ = 1e-6
LR = LogisticRegression()

dataset_LDA = LDA(dataset, labels, 7)
evaluation_dataset_LDA = LDA(evaluation_dataset, evaluation_labels, 7)
LR.train(dataset_LDA, labels, lambda_)
print(f"LR Error Rate: {utils.computeErrorRate(LR.predict(evaluation_dataset_LDA), evaluation_labels)}")
print(f"LR MinDCF: {utils.minDCF(LR, evaluation_dataset_LDA, evaluation_labels, prior, Cfn, Cfp)}\n")

############### MVG (PCA m=8) ###############
MVG = GaussianClassifierFullCovariance()

MVG.train(dataset, labels)
print(f"MVG Error Rate: {utils.computeErrorRate(MVG.predict(evaluation_dataset), evaluation_labels)}")
print(f"MVG MinDCF: {utils.minDCF(MVG, evaluation_dataset, evaluation_labels, prior, Cfn, Cfp)}\n")

############### Radial Basis Function SVM ###############
C = 12.915496650148826
gamma = 0.001
K = 10

RBF_SVM = RadialBasisFunctionSVM()
RBF_SVM.train(dataset_svm, labels_svm, C, gamma, K)
print(f"RBF_SVM Error Rate: {utils.computeErrorRate(RBF_SVM.predict(evaluation_dataset_svm), evaluation_labels_svm)}")
print(f"RBF_SVM MinDCF: {utils.minDCF(RBF_SVM, evaluation_dataset_svm, evaluation_labels_svm, prior, Cfn, Cfp)}\n")

############### GMM 16 components ###############
GMM = GMMFullCovariance()

components = 4
GMM.train(dataset, labels, components)
#print(f"GMM Error Rate: {utils.computeErrorRate(GMM.predict(evaluation_dataset), evaluation_labels)}")
print(f"GMM MinDCF: {utils.minDCF(GMM, evaluation_dataset, evaluation_labels, prior, Cfn, Cfp)}\n")

print("Evaluation finished.")

