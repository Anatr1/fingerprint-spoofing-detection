import numpy as np
import utils
import plots
from SVM import RadialBasisFunctionSVM
from GMM import GMMFullCovariance
from MVG import GaussianClassifierFullCovariance

if __name__ == "__main__":
    # Data Loading
    train_dataset, train_labels = utils.loadData("../data/Train.txt")
    test_dataset, test_labels = utils.loadData("../data/Test.txt")
    train_dataset_svm, train_labels_svm = utils.loadDataChangeLabels("../data/Train.txt")
    test_dataset_svm, test_labels_svm = utils.loadDataChangeLabels("../data/Test.txt")

    prior = 0.5
    minDCF_MVG = 0.288
    minDCF_GMM = 0.138
    minDCF_SVM = 0.238
    C = 12.915496650148826
    gamma = 0.001
    K = 10
    lambda_calibration = 1e-3

    MVG = GaussianClassifierFullCovariance()
    GMM = GMMFullCovariance()
    SVM = RadialBasisFunctionSVM()

    print("MVG Full Covariance")
    FPR_MVG = []
    TPR_MVG = []
    MVG.train(train_dataset, train_labels)
    uncalibrated_scores = MVG.getScores(test_dataset)
    scores = utils.calibrateScores(uncalibrated_scores, test_labels, lambda_calibration, prior)[0]
    sorted_scores=np.sort(scores)
    for t in sorted_scores:
        confusion_matrix = utils.optimalBayesConfusionMatrix(scores, test_labels, t)
        FPRtemp, TPRtemp = utils.computeFPRTPR(confusion_matrix)
        FPR_MVG.append(FPRtemp)
        TPR_MVG.append(TPRtemp)
    print("End MVG Tied-Cov")

    print("GMM Full Covariance")
    FPR_GMM = []
    TPR_GMM = []
    GMM.train(train_dataset, train_labels, 4)
    uncalibrated_scores = GMM.getScores(test_dataset)
    scores = utils.calibrateScores(uncalibrated_scores, test_labels, lambda_calibration, prior)[0]
    sorted_scores=np.sort(scores)
    for t in sorted_scores:
        confusion_matrix = utils.optimalBayesConfusionMatrix(scores, test_labels, t)
        FPRtemp, TPRtemp = utils.computeFPRTPR(confusion_matrix)
        FPR_GMM.append(FPRtemp)
        TPR_GMM.append(TPRtemp)
    print("End GMM")

    print("RBF SVM")
    FPR_SVM = []
    TPR_SVM = []
    SVM.train(train_dataset_svm, train_labels_svm, C, gamma, K)
    uncalibrated_scores = SVM.getScores(test_dataset_svm)
    scores = utils.calibrateScores(uncalibrated_scores, test_labels_svm, lambda_calibration, prior, True)[0]
    sorted_scores=np.sort(scores)
    for t in sorted_scores:
        confusion_matrix = utils.optimalBayesConfusionMatrix(scores, test_labels_svm, t)
        FPRtemp, TPRtemp = utils.computeFPRTPR(confusion_matrix)
        FPR_SVM.append(FPRtemp)
        TPR_SVM.append(TPRtemp)
    print("End Linear SVM")

    # Plot ROC
    plots.plotROC(FPR_MVG, TPR_MVG, FPR_GMM, TPR_GMM, FPR_SVM, TPR_SVM)