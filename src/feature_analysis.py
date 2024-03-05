# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file plots the histograms and scatters of the dataset.

import matplotlib.pyplot as plt
import utils
from PCA import PCA
from LDA import LDA
from plots import plotScatters, plotHistograms, plotCorrelationHeatmap, plotPCA, savePlotPCA, plotPie

dataset, labels = utils.loadData("../data/Train.txt")
evaluation_dataset, evaluation_labels = utils.loadData("../data/Test.txt")
(training_dataset, training_labels), (cross_validation_dataset, cross_validation_labels) = utils.singleFold(dataset, labels)

################# PCA & LDA #####################
print(dataset.shape)
plotHistograms(dataset, labels)  
plotScatters(dataset, labels)  
plt.show()

dataset_PCA = PCA(dataset, 9)
print(dataset_PCA.shape)
plotHistograms(dataset_PCA, labels)
plotScatters(dataset_PCA, labels) 
plt.show()

dataset_PCA = PCA(dataset, 8)
print(dataset_PCA.shape)
plotHistograms(dataset_PCA, labels)  
plotScatters(dataset_PCA, labels)  
plt.show()

dataset_PCA = PCA(dataset, 7)
print(dataset_PCA.shape)
plotHistograms(dataset_PCA, labels)   
plotScatters(dataset_PCA, labels)
plt.show()

dataset_PCA = PCA(dataset, 6)
print(dataset_PCA.shape)
plotHistograms(dataset_PCA, labels)    
plotScatters(dataset_PCA, labels)
plt.show()

dataset_PCA = PCA(dataset, 5)
print(dataset_PCA.shape)
plotHistograms(dataset_PCA, labels)    
plotScatters(dataset_PCA, labels)
plt.show()

dataset_PCA = PCA(dataset, 4)
print(dataset_PCA.shape)
plotHistograms(dataset_PCA, labels)    
plotScatters(dataset_PCA, labels)
plt.show()

dataset_LDA = LDA(dataset, labels, 9)
print(dataset_LDA.shape)
plotHistograms(dataset_LDA, labels)
plotScatters(dataset_LDA, labels)
plt.show()

dataset_LDA = LDA(dataset, labels, 8)
print(dataset_LDA.shape)
plotHistograms(dataset_LDA, labels)
plotScatters(dataset_LDA, labels)
plt.show()

dataset_LDA = LDA(dataset, labels, 7)
print(dataset_LDA.shape)
plotHistograms(dataset_LDA, labels)
plotScatters(dataset_LDA, labels)
plt.show()

dataset_LDA = LDA(dataset, labels, 6)
print(dataset_LDA.shape)
plotHistograms(dataset_LDA, labels)
plotScatters(dataset_LDA, labels)
plt.show()

dataset_LDA = LDA(dataset, labels, 5)
print(dataset_LDA.shape)
plotHistograms(dataset_LDA, labels)
plotScatters(dataset_LDA, labels)
plt.show()

dataset_LDA = LDA(dataset, labels, 4)
print(dataset_LDA.shape)
plotHistograms(dataset_LDA, labels)
plotScatters(dataset_LDA, labels)
plt.show()

plotCorrelationHeatmap(dataset, labels)
plt.show()

plotPCA(dataset)
savePlotPCA(dataset, "../plots/PCA_testsave.png")
plotCorrelationHeatmap(dataset, labels)
plotCorrelationHeatmap(evaluation_dataset, evaluation_labels)

plotPie(labels, ["Spoofed", "Authentic"])
plotPie(evaluation_labels, ["Spoofed", "Authentic"])
