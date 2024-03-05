import numpy as np
import utils
import plots
from LDA import LDA
from PCA import PCA
from SVM import RadialBasisFunctionSVM
from LR  import LogisticRegression
from MVG import GaussianClassifierFullCovariance
from GMM import GMMFullCovariance
from CrossValidation import KFoldCrossValidation

N_FOLDS = 6
N_POINTS = 25

if __name__ == "__main__":
    # Data Loading
    train_dataset, train_labels = utils.loadData("../data/Train.txt")
    train_dataset_svm, train_labels_svm = utils.loadDataChangeLabels("../data/Train.txt")
    (cross_training_dataset_svm, cross_training_labels_svm), (cross_validation_dataset_svm, cross_validation_labels_svm) = utils.singleFold(train_dataset_svm, train_labels_svm)
    (cross_training_dataset_gmm, cross_training_labels_gmm), (cross_validation_dataset_gmm, cross_validation_labels_gmm) = utils.singleFold(train_dataset, train_labels)

    prior = 0.5
    Cfn = 1
    Cfp = 10

    effPriorLog = np.linspace(-3, 3, N_POINTS)
    effPriors = 1/(1+np.exp(-1*effPriorLog))

    # If scores are already well calibrated, the optimal threshold that optimizes the Bayes Risk is t = -log(p/1-p)

    # Step 1:
    # Step 1.1: Assume scores are well calibrated, compute actual Detection Cost Function (ActDCF) and Minimum Detection Cost Function (MinDCF) using theoritical threshold t = -log(p/1-p)

    # MinDCF for MVG Full-Cov (PCA (m=8)) = 0.288. Error Rate = 5.68%
    minDCF_MVG = 0.288
    # MinDCF for GMM K=16 = 0.138. Error Rate = 2.57%
    minDCF_GMM = 0.238
    # MinDCF for RBF SVM (gamma = 0.001, K = 10, C = 12.915496650148826, NO PCA) = 0.297. Error Rate = 4.77%
    gamma = 0.001
    K = 10
    C = 12.915496650148826
    minDCF_RBFSVM = 0.297
    
    # Multivariate Gaussian Classifier Tied-Covariance (PCA (m=8))
    gaussian_classifier_full = GaussianClassifierFullCovariance()
    MVG_train_dataset = PCA(train_dataset, 8)
    print("Actual DCF for MVG Full-Covariance (PCA (m=7))")
    _, _, MVG_actual_DCF, _ = KFoldCrossValidation(gaussian_classifier_full, MVG_train_dataset, train_labels, k=N_FOLDS, DIMENSIONS=8, prior=prior, Cfn=Cfn, Cfp=Cfp, actualDCF=True)
    _, _, MVG_min_DCF, _ = KFoldCrossValidation(gaussian_classifier_full, MVG_train_dataset, train_labels, k=N_FOLDS, DIMENSIONS=8, prior=prior, Cfn=Cfn, Cfp=Cfp, actualDCF=False)
    print(f"Prior = {prior} Min DCF = {round(MVG_min_DCF,3)} Actual DCF =  {round(MVG_actual_DCF, 3)}\n")

    # RBF SVM (NO PCA)
    RFB_SVM_classifier = RadialBasisFunctionSVM()
    print("Actual DCF for RBF SVM (NO PCA)")
    _, _, SVM_actual_DCF, _ = KFoldCrossValidation(RFB_SVM_classifier, train_dataset_svm, train_labels_svm, k=N_FOLDS, SVM=['RBF', C, gamma, K], prior=prior, Cfn=Cfn, Cfp=Cfp, actualDCF=True)
    _, _, SVM_min_DCF, _ = KFoldCrossValidation(RFB_SVM_classifier, train_dataset_svm, train_labels_svm, k=N_FOLDS, SVM=['RBF', C, gamma, K], prior=prior, Cfn=Cfn, Cfp=Cfp, actualDCF=False)
    print(f"Prior = {prior} Min DCF = {round(SVM_min_DCF,3)} Actual DCF =  {round(SVM_actual_DCF, 3)}\n")
    
    # GMM Full-Covariance (K=16)
    gmm_classifier_full = GMMFullCovariance()
    print("Actual DCF for GMM Full-Covariance (K=16)")
    _, _, GMM_actual_DCF, _ = KFoldCrossValidation(gmm_classifier_full, train_dataset, train_labels, k=N_FOLDS, components=16, num_classes=2, prior=prior, Cfn=Cfn, Cfp=Cfp, actualDCF=True)
    _, _, GMM_min_DCF, _ = KFoldCrossValidation(gmm_classifier_full, train_dataset, train_labels, k=N_FOLDS, components=16, num_classes=2, prior=prior, Cfn=Cfn, Cfp=Cfp, actualDCF=False)
    print(f"Prior = {prior} Min DCF = {round(GMM_min_DCF,3)} Actual DCF =  {round(GMM_actual_DCF, 3)}\n")
    
     
    # Step 1.2: Check gap between ActDCF and MinDCF, if gap is small, then scores are well calibrated. 
    #           --> They are not well calibrated

    # Step 1.3: Plot Bayes Error Plot for each model for both ActDCF and MinDCF, if the plot is similar, then scores are well calibrated.

    # MVG Full-Covariance (PCA (m=8)) Bayes Error Plot
    actual_DCFs = []
    min_DCFs = []
    for i in range(N_POINTS):
        actual_DCF = utils.calibrate(gaussian_classifier_full, MVG_train_dataset, train_labels, effPriors[i], Cfn, Cfp, K=N_FOLDS, actualDCF=True, calibration=False)
        min_DCF = utils.calibrate(gaussian_classifier_full, MVG_train_dataset, train_labels, effPriors[i], Cfn, Cfp, K=N_FOLDS, actualDCF=False, calibration=False)
        actual_DCFs.append(actual_DCF)
        min_DCFs.append(min_DCF)
        print(f"Point {i+1}/{N_POINTS}: Actual DCF = {round(actual_DCF,3)} Min DCF = {round(min_DCF,3)}")
    plots.plotBayesError(actual_DCFs, min_DCFs, effPriorLog, "MVG Full-Covariance (PCA (m=8))")
    
    # RBF SVM (NO PCA) Bayes Error Plot
    actual_DCFs = []
    min_DCFs = []
    RFB_SVM_classifier.train(cross_training_dataset_svm, cross_training_labels_svm, C, gamma, K)
    for i in range(N_POINTS):
        actual_DCFs.append(utils.actDCFFromScores(effPriors[i], 1, 1, RFB_SVM_classifier.getScores(cross_validation_dataset_svm), cross_validation_labels_svm))
        min_DCFs.append(utils.minDCFFromScores(effPriors[i], 1, 1, RFB_SVM_classifier.getScores(cross_validation_dataset_svm), cross_validation_labels_svm))
        print(f"Point {i+1}/{N_POINTS}: Actual DCF = {round(actual_DCFs[i],3)} Min DCF = {round(min_DCFs[i],3)}")
    plots.plotBayesError(actual_DCFs, min_DCFs, effPriorLog, "RBF SVM (NO PCA)")

    
    # GM Full-Covariance (K=16) Bayes Error Plot
    actual_DCFs = []
    min_DCFs = []
    gmm_classifier_full.train(cross_training_dataset_gmm, cross_training_labels_gmm, 16)
    for i in range(N_POINTS):
        actual_DCFs.append(utils.actDCFFromScores(effPriors[i], 1, 1, gmm_classifier_full.getScores(cross_validation_dataset_gmm), cross_validation_labels_gmm))
        min_DCFs.append(utils.minDCFFromScores(effPriors[i], 1, 1, gmm_classifier_full.getScores(cross_validation_dataset_gmm), cross_validation_labels_gmm))
        print(f"Point {i+1}/{N_POINTS}: Actual DCF = {round(actual_DCFs[i],3)} Min DCF = {round(min_DCFs[i],3)}")
    plots.plotBayesError(actual_DCFs, min_DCFs, effPriorLog, "GMM Full-Covariance (K=16)")

    # Step 2:
    # Step 2.1: Compute the transformation function that maps the scores to the new scores scal = f(s) = alpha * s + beta => f(s) = alpha * s + beta' - log(p/1-p) 
    #           --> Is the same as logistic regression to train to learn parameters alpha and beta'

    # Step 2.2: Plot Bayes Error Plot for each model for different values of lambda, if the plot is similar, then scores are well calibrated.

    # Calibrate MVG Full-Covariance (PCA (m=8))
    actual_DCFs1 = []
    min_DCFs = []
    for i in range(N_POINTS):
        actual_DCFs1.append(utils.calibrate(gaussian_classifier_full, MVG_train_dataset, train_labels, effPriors[i], Cfn, Cfp, N_FOLDS, actualDCF=True, lambda_calibration= 1e-3))
        min_DCFs.append(utils.calibrate(gaussian_classifier_full, MVG_train_dataset, train_labels, effPriors[i], Cfn, Cfp, N_FOLDS, actualDCF=False))
        print(f"Point {i+1}/{N_POINTS}: Actual DCF = {round(actual_DCFs1[i],3)} Min DCF = {round(min_DCFs[i],3)}")
    plots.plotBayesErrorCalibrated(actual_DCFs1, min_DCFs, effPriorLog, "MVG Full-Covariance (PCA (m=8))", 1e-3)

    # Calibrate RBF SVM (NO PCA)
    actual_DCFs1 = []
    min_DCFs = []
    RFB_SVM_classifier.train(cross_training_dataset_svm, cross_training_labels_svm, C, gamma, K)
    for i in range(N_POINTS):
        actual_DCFs1.append(utils.actDCFFromScores(effPriors[i], 1, 1, utils.calibrateScores(RFB_SVM_classifier.getScores(cross_validation_dataset_svm), cross_validation_labels_svm, 1e-3, 0.5, SVM=True)[0], cross_validation_labels_svm))
        min_DCFs.append(utils.minDCFFromScores(effPriors[i], 1, 1, RFB_SVM_classifier.getScores(cross_validation_dataset_svm), cross_validation_labels_svm))
        print(f"Point {i+1}/{N_POINTS}: Actual DCF = {round(actual_DCFs1[i],3)} Min DCF = {round(min_DCFs[i],3)}")
    plots.plotBayesErrorCalibrated(actual_DCFs1, min_DCFs, effPriorLog, "RBF SVM (NO PCA)", 1e-3)

    # Calibrate GMM Full-Covariance (K=16)
    actual_DCFs1 = []
    min_DCFs = []
    gmm_classifier_full.train(cross_training_dataset_gmm, cross_training_labels_gmm, 16)
    for i in range(N_POINTS):
        actual_DCFs1.append(utils.actDCFFromScores(effPriors[i], 1, 1, utils.calibrateScores(gmm_classifier_full.getScores(cross_validation_dataset_gmm), cross_validation_labels_gmm, 1e-3, 0.5)[0], cross_validation_labels_gmm))
        min_DCFs.append(utils.minDCFFromScores(effPriors[i], 1, 1, gmm_classifier_full.getScores(cross_validation_dataset_gmm), cross_validation_labels_gmm))
        print(f"Point {i+1}/{N_POINTS}: Actual DCF = {round(actual_DCFs1[i],3)} Min DCF = {round(min_DCFs[i],3)}")
    plots.plotBayesErrorCalibrated(actual_DCFs1, min_DCFs, effPriorLog, "GMM Full-Covariance (K=16)", 1e-3)
