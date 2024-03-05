# Machine Learning and Pattern Recognition Project
# Fingerprint Spoofing Detection
# Author: Federico Mustich
# Date: 01/2024
# Brief: This file contains the functions to plot the results of the experiments.
import utils
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn
import numpy as np
from PCA import getEigen


features = []
classes = ["Spoofed", "Authentic"]

def plotPie(labels, legend):
    # Count the number of 0s and 1s
    counts = [np.count_nonzero(labels==i) for i in np.unique(labels)]
    
    # Specify colors
    colors = ['cyan', 'magenta']
    
    # Plot the pie chart
    plt.pie(counts, labels=legend, colors=colors, autopct='%1.1f%%')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

def plotHistogram(feature, dataset, labels, classes):
    plt.hist(
        dataset[feature, labels == 0],
        color="cyan",
        density=True,
        alpha=0.6,
        bins=50,
    )
    plt.hist(
        dataset[feature, labels == 1],
        color="magenta",
        density=True,
        alpha=0.4,
        bins=50,
    )
    plt.legend(classes)
    #plt.show()

def plotHistograms(dataset, labels):
    plt.figure()
    for i in range(dataset.shape[0]):
        plt.subplot(5, 2, i + 1)
        plotHistogram(i, dataset, labels, classes)

def plotScatter(i, j, dataset, labels, classes):
    plt.scatter(dataset[i, labels == 0], dataset[j, labels == 0], color="cyan", s=2)
    plt.scatter(dataset[i, labels == 1], dataset[j, labels == 1], color="magenta", s=2)
    #plt.legend(classes)
    #plt.show()

def plotScatters(dataset, labels):
    plt.figure()
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            if i != j:
                plt.subplot(10, 10, i * 10 + j + 1)
                plotScatter(i, j, dataset, labels, classes)

def plotCorrelationHeatmap(dataset, labels):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))  # Create 3 subplots arranged horizontally

    seaborn.heatmap(
        np.corrcoef(dataset),
        linewidth=0.2,
        cmap=seaborn.light_palette("gray", as_cmap=True),
        square=True,
        cbar=False,
        annot=True,
        ax=axs[0]  # Plot on the first subplot
    )
    axs[0].set_title('All data')

    seaborn.heatmap(
        np.corrcoef(dataset[:, labels == 1]),
        linewidth=0.2,
        cmap=seaborn.light_palette("magenta", as_cmap=True),
        square=True,
        cbar=False,
        annot=True,
        ax=axs[1]  # Plot on the second subplot
    )
    axs[1].set_title('Authentic')

    seaborn.heatmap(
        np.corrcoef(dataset[:, labels == 0]),
        linewidth=0.2,
        cmap=seaborn.light_palette("cyan", as_cmap=True),
        square=True,
        cbar=False,
        annot=True,
        ax=axs[2]  # Plot on the third subplot
    )
    axs[2].set_title('Spoofed')

    plt.tight_layout()
    plt.show()

def plotPCA(dataset):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = getEigen(dataset)

    # Total sum of eigenvalues
    total = eigenvalues.sum()

    # List to store information retrieved
    info_retrieved = []

    for m in range(1, dataset.shape[0] + 1):
        # Compute information retrieved
        info = eigenvalues[:m].sum() / total
        info_retrieved.append(info * 100)

    # Plot information retrieved
    plt.plot(range(1, dataset.shape[0] + 1), info_retrieved)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Information Retrieved')
    plt.title('Information Retrieved by PCA')

    # Set grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Major grid lines
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  # Minor grid lines
    plt.minorticks_on()  # Enable minor ticks

    plt.show()

def savePlotPCA(dataset, filename):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = getEigen(dataset)

    # Total sum of eigenvalues
    total = eigenvalues.sum()

    # List to store information retrieved
    info_retrieved = []

    for m in range(1, dataset.shape[0] + 1):
        # Compute information retrieved
        info = eigenvalues[:m].sum() / total
        info_retrieved.append(info * 100)

    # Plot information retrieved
    plt.plot(range(1, dataset.shape[0] + 1), info_retrieved)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Information Retrieved')
    plt.title('Information Retrieved by PCA')

    # Set grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Major grid lines
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  # Minor grid lines
    plt.minorticks_on()  # Enable minor ticks

    # Save plot at maximum resolution
    plt.savefig(filename, dpi=300)

def plotMVG(results, error_rate=False):
    # Define the labels for the x-axis
    #x_labels = [str(10 - i) for i in range(len(results))]
    x_labels = ["No Preprocessing", "PCA (m=9)", "PCA (m=8)", "PCA (m=7)", "LDA (m=9)", "LDA (m=8)", "LDA (m=7)"]
    
    if error_rate:
        y_label = "Error Rate"
    else:
        y_label = "minDCF"

    # Define the width of the bars
    bar_width = 0.15

    # Define the x-coordinates of the bars for each classifier
    x_FC = np.arange(len(x_labels))
    x_NB = [x + bar_width for x in x_FC]
    x_FTC = [x + bar_width for x in x_NB]
    x_NBT = [x + bar_width for x in x_FTC]

    # Extract the results for each classifier
    results_FC  = [results[i][0] for i in range(len(results))]
    results_NB  = [results[i][1] for i in range(len(results))]
    results_FTC = [results[i][2] for i in range(len(results))]
    results_NBT = [results[i][3] for i in range(len(results))]

    # Create the bar plots
    plt.bar(x_FC, results_FC, color='#ff6f61', width=bar_width, label='Full Covariance')
    plt.bar(x_NB, results_NB, color='#6b5b95', width=bar_width, label='Diagonal Covariance')
    plt.bar(x_FTC, results_FTC, color='#feb236', width=bar_width, label='Tied Full Covariance')
    plt.bar(x_NBT, results_NBT, color='#d64161', width=bar_width, label='Tied Diagonal Covariance')

    # Add labels and title
    #plt.xlabel('Number of Principal Components')
    plt.ylabel(y_label)
    plt.title(f'{y_label} for Multivariate Gaussian Classifiers')

    # Add xticks on the middle of the group bars
    plt.xticks([r + bar_width for r in range(len(x_labels))], x_labels)

    # Set grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Major grid lines
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  # Minor grid lines
    plt.minorticks_on()  # Enable minor ticks

    # Create legend
    plt.legend()

    # Show plot
    plt.show()

def plotKNN(x, y, error_rate=False):
    if error_rate:
        y_label = "Error Rate"
    else:
        y_label = "DCF"
    colors = ["b", "r", "g", "y", "c", "orange", "m"]
    labels = ["No Preprocessing", "PCA (m=9)", "PCA (m=8)", "PCA (m=7)", "LDA (m=9)", "LDA (m=8)", "LDA (m=7)"]

    plt.figure()
    for i in range(len(labels)):
        #if len(y[i * len(x) : (i+1) * len(x)]) == len(x):
        plt.plot(x, y[i * len(x) : (i+1) * len(x)], label=labels[i], color=colors[i])
    
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(labels)
    plt.xlabel("k")
    plt.ylabel(y_label)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Major grid lines
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  # Minor grid lines
    plt.minorticks_on()  # Enable minor ticks
    plt.show()

def plotLambda(x, y):
    plt.figure()
    plt.plot(x, y[0 : len(x)], label="No Preprocessing", color="b")
    plt.plot(x, y[len(x) : 2 * len(x)], label="PCA (m=9)", color="r")
    plt.plot(x, y[2 * len(x) : 3 * len(x)], label="PCA (m=8)", color="g")
    plt.plot(x, y[3 * len(x) : 4 * len(x)], label="PCA (m=7)", color="y")
    plt.plot(x, y[4 * len(x) : 5 * len(x)], label="LDA (m=9)", color="k")
    plt.plot(x, y[5 * len(x) : 6 * len(x)], label="LDA (m=8)", color="#8c564b")
    plt.plot(x, y[6 * len(x) : 7 * len(x)], label="LDA (m=7)", color="#e377c2")
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["No Preprocessing", "PCA (m=9)", "PCA (m=8)", "PCA (m=7)", "LDA (m=9)", "LDA (m=8)", "LDA (m=7)"])
    plt.xlabel("λ")
    plt.ylabel("minDCF")
    plt.show()

def plotLambdaQLR(x, y):
    plt.figure()
    plt.plot(x, y[0 : len(x)], label="No Preprocessing", color="b")
    plt.plot(x, y[len(x) : 2 * len(x)], label="PCA (m=9)", color="r")
    plt.plot(x, y[2 * len(x) : 3 * len(x)], label="LDA (m=9)", color="k")
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["No Preprocessing", "PCA (m=9)", "LDA (m=9)"])
    plt.xlabel("λ")
    plt.ylabel("minDCF")
    plt.show()

def plotLinearSVM(x, y, error_rate=False):
    if error_rate:
        y_label = "Error Rate"
    else:
        y_label = "minDCF"

    plt.figure(figsize=(10, 6)) 
    plt.plot(x, y[0 : len(x)], label="No Preprocessing, K = 0.01", color="b")
    plt.plot(x, y[len(x) : 2 * len(x)], label="No Preprocessing, K = 0.1", color="r")
    plt.plot(x, y[2 * len(x) : 3 * len(x)], label="No Preprocessing, K = 1", color="g")
    plt.plot(x, y[3 * len(x) : 4 * len(x)], label="No Preprocessing, K = 10", color="y")
    plt.plot(x, y[4 * len(x) : 5 * len(x)], label="PCA (m=9), K = 0.01", color="c")
    plt.plot(x, y[5 * len(x) : 6 * len(x)], label="PCA (m=9), K = 0.1", color="m")
    plt.plot(x, y[6 * len(x) : 7 * len(x)], label="PCA (m=9), K = 1", color="k")
    plt.plot(x, y[7 * len(x) : 8 * len(x)], label="PCA (m=9), K = 10", color="#8c564b")
    plt.plot(x, y[8 * len(x) : 9 * len(x)], label="PCA (m=8), K = 0.01", color="#e377c2")
    plt.plot(x, y[9 * len(x) : 10 * len(x)], label="PCA (m=8), K = 0.1", color="#7f7f7f")
    plt.plot(x, y[10 * len(x) : 11 * len(x)], label="PCA (m=8), K = 1", color="#bcbd66")
    plt.plot(x, y[11 * len(x) : 12 * len(x)], label="PCA (m=8), K = 10", color="#17becf")
    plt.plot(x, y[12 * len(x) : 13 * len(x)], label="LDA (m=9), K = 0.01", color="#9467bd")
    plt.plot(x, y[13 * len(x) : 14 * len(x)], label="LDA (m=9), K = 0.1", color="#d62728")
    plt.plot(x, y[14 * len(x) : 15 * len(x)], label="LDA (m=9), K = 1", color="#8cf94b")
    plt.plot(x, y[15 * len(x) : 16 * len(x)], label="LDA (m=9), K = 10", color="#e444c2")
    plt.plot(x, y[16 * len(x) : 17 * len(x)], label="LDA (m=8), K = 0.01", color="#8f8f8f")
    plt.plot(x, y[17 * len(x) : 18 * len(x)], label="LDA (m=8), K = 0.1", color="#bcbd22")
    plt.plot(x, y[18 * len(x) : 19 * len(x)], label="LDA (m=8), K = 1", color="#98ddcf")
    plt.plot(x, y[19 * len(x) : 20 * len(x)], label="LDA (m=8), K = 10", color="#8365bd")
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["No Preprocessing, K = 0.01", "No Preprocessing, K = 0.1", "No Preprocessing, K = 1", "No Preprocessing, K = 10", "PCA (m=9), K = 0.01", "PCA (m=9), K = 0.1", "PCA (m=9), K = 1", "PCA (m=9), K = 10", "PCA (m=8), K = 0.01", "PCA (m=8), K = 0.1", "PCA (m=8), K = 1", "PCA (m=8), K = 10", "LDA (m=9), K = 0.01", "LDA (m=9), K = 0.1", "LDA (m=9), K = 1", "LDA (m=9), K = 10", "LDA (m=8), K = 0.01", "LDA (m=8), K = 0.1", "LDA (m=8), K = 1", "LDA (m=8), K = 10"], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=5)
    plt.xlabel("C")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

def plotPolySVM(x, y, error_rate=False):
    if error_rate:
        y_label = "Error Rate"
    else:
        y_label = "minDCF"

    plt.figure(figsize=(7, 6))
    plt.plot(x, y[0 : len(x)], label="No Preprocessing, c = 0, d = 2, K = 1", color="b")
    plt.plot(x, y[len(x) : 2 * len(x)], label="No Preprocessing, c = 0, d = 2, K = 10", color="r")
    plt.plot(x, y[2 * len(x) : 3 * len(x)], label="No Preprocessing, c = 0, d = 3, K = 1", color="g")
    plt.plot(x, y[3 * len(x) : 4 * len(x)], label="No Preprocessing, c = 0, d = 3, K = 10", color="y")
    plt.plot(x, y[4 * len(x) : 5 * len(x)], label="No Preprocessing, c = 1, d = 2, K = 1", color="c")
    plt.plot(x, y[5 * len(x) : 6 * len(x)], label="No Preprocessing, c = 1, d = 2, K = 10", color="m")
    plt.plot(x, y[6 * len(x) : 7 * len(x)], label="No Preprocessing, c = 1, d = 3, K = 1", color="k")
    plt.plot(x, y[7 * len(x) : 8 * len(x)], label="No Preprocessing, c = 1, d = 3, K = 10", color="#8c564b")
    plt.plot(x, y[8 * len(x) : 9 * len(x)], label="PCA (m=9), c = 0, d = 2, K = 1", color="#e377c2")
    plt.plot(x, y[9 * len(x) : 10 * len(x)], label="PCA (m=9), c = 0, d = 2, K = 10", color="#7f7f7f")
    plt.plot(x, y[10 * len(x) : 11 * len(x)], label="PCA (m=9), c = 0, d = 3, K = 1", color="#bcbd66")
    plt.plot(x, y[11 * len(x) : 12 * len(x)], label="PCA (m=9), c = 0, d = 3, K = 10", color="#17becf")
    plt.plot(x, y[12 * len(x) : 13 * len(x)], label="PCA (m=9), c = 1, d = 2, K = 1", color="#9467bd") 
    plt.plot(x, y[13 * len(x) : 14 * len(x)], label="PCA (m=9), c = 1, d = 2, K = 10", color="#d62728")
    plt.plot(x, y[14 * len(x) : 15 * len(x)], label="PCA (m=9), c = 1, d = 3, K = 1", color="#8cf94b")
    plt.plot(x, y[15 * len(x) : 16 * len(x)], label="PCA (m=9), c = 1, d = 3, K = 10", color="#e444c2")
    plt.plot(x, y[16 * len(x) : 17 * len(x)], label="LDA (m=9), c = 0, d = 2, K = 1", color="#8f8f8f")
    plt.plot(x, y[17 * len(x) : 18 * len(x)], label="LDA (m=9), c = 0, d = 2, K = 10", color="#bcbd22")
    plt.plot(x, y[18 * len(x) : 19 * len(x)], label="LDA (m=9), c = 0, d = 3, K = 1", color="#98ddcf")
    plt.plot(x, y[19 * len(x) : 20 * len(x)], label="LDA (m=9), c = 0, d = 3, K = 10", color="#8365bd")
    plt.plot(x, y[20 * len(x) : 21 * len(x)], label="LDA (m=9), c = 1, d = 2, K = 1", color="#ff6f61")
    plt.plot(x, y[21 * len(x) : 22 * len(x)], label="LDA (m=9), c = 1, d = 2, K = 10", color="#6b5b95")
    plt.plot(x, y[22 * len(x) : 23 * len(x)], label="LDA (m=9), c = 1, d = 3, K = 1", color="#feb236")
    plt.plot(x, y[23 * len(x) : 24 * len(x)], label="LDA (m=9), c = 1, d = 3, K = 10", color="#d64161")
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["No Pre c=0 d=2 K=1", "No Pre c=0 d=2 K=10", "No Pre c=0 d=3 K=1", "No Pre c=0 d=3 K=10", "No Pre c=1 d=2 K=1", "No Pre c=1 d=2 K=10", "No Pre c=1 d=3 K=1", "No Pre c=1 d=3 K=10", "PCA m=9 c=0 d=2 K=1", "PCA m=9 c=0 d=2 K=10", "PCA m=9 c=0 d=3 K=1", "PCA m=9 c=0 d=3 K=10", "PCA m=9 c=1 d=2 K=1", "PCA m=9 c=1 d=2 K=10", "PCA m=9 c=1 d=3 K=1", "PCA m=9 c=1 d=3 K=10", "LDA m=9 c=0 d=2 K=1", "LDA m=9 c=0 d=2 K=10", "LDA m=9 c=0 d=3 K=1", "LDA m=9 c=0 d=3 K=10", "LDA m=9 c=1 d=2 K=1", "LDA m=9 c=1 d=2 K=10", "LDA m=9 c=1 d=3 K=1", "LDA m=9 c=1 d=3 K=10"], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=3)
    plt.xlabel("C")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

def plotRBFSVM(x, y, error_rate=False):
    if error_rate:
        y_label = "Error Rate"
    else:
        y_label = "minDCF"

    plt.figure(figsize=(7, 6))
    plt.plot(x, y[0 : len(x)], label="No Preprocessing, γ = 0.0001, K = 1", color="b")
    plt.plot(x, y[len(x) : 2 * len(x)], label="No Preprocessing, γ = 0.0001, K = 10", color="r")
    plt.plot(x, y[2 * len(x) : 3 * len(x)], label="No Preprocessing, γ = 0.001, K = 1", color="g")
    plt.plot(x, y[3 * len(x) : 4 * len(x)], label="No Preprocessing, γ = 0.001, K = 10", color="y")
    plt.plot(x, y[4 * len(x) : 5 * len(x)], label="No Preprocessing, γ = 0.01, K = 1", color="c")
    plt.plot(x, y[5 * len(x) : 6 * len(x)], label="No Preprocessing, γ = 0.01, K = 10", color="m")
    plt.plot(x, y[6 * len(x) : 7 * len(x)], label="No Preprocessing, γ = 0.1, K = 1", color="k")
    plt.plot(x, y[7 * len(x) : 8 * len(x)], label="No Preprocessing, γ = 0.1, K = 10", color="#8c564b")
    plt.plot(x, y[8 * len(x) : 9 * len(x)], label="PCA (m=9), γ = 0.0001, K = 1", color="#e377c2")
    plt.plot(x, y[9 * len(x) : 10 * len(x)], label="PCA (m=9), γ = 0.0001, K = 10", color="#7f7f7f")
    plt.plot(x, y[10 * len(x) : 11 * len(x)], label="PCA (m=9), γ = 0.001, K = 1", color="#bcbd66")
    plt.plot(x, y[11 * len(x) : 12 * len(x)], label="PCA (m=9), γ = 0.001, K = 10", color="#17becf")
    plt.plot(x, y[12 * len(x) : 13 * len(x)], label="PCA (m=9), γ = 0.01, K = 1", color="#9467bd")
    plt.plot(x, y[13 * len(x) : 14 * len(x)], label="PCA (m=9), γ = 0.01, K = 10", color="#d62728")
    plt.plot(x, y[14 * len(x) : 15 * len(x)], label="PCA (m=9), γ = 0.1, K = 1", color="#8cf94b")
    plt.plot(x, y[15 * len(x) : 16 * len(x)], label="PCA (m=9), γ = 0.1, K = 10", color="#e444c2")
    plt.plot(x, y[16 * len(x) : 17 * len(x)], label="LDA (m=9), γ = 0.0001, K = 1", color="#8f8f8f")
    plt.plot(x, y[17 * len(x) : 18 * len(x)], label="LDA (m=9), γ = 0.0001, K = 10", color="#bcbd22")
    plt.plot(x, y[18 * len(x) : 19 * len(x)], label="LDA (m=9), γ = 0.001, K = 1", color="#98ddcf")
    plt.plot(x, y[19 * len(x) : 20 * len(x)], label="LDA (m=9), γ = 0.001, K = 10", color="#8365bd")
    plt.plot(x, y[20 * len(x) : 21 * len(x)], label="LDA (m=9), γ = 0.01, K = 1", color="#ff6f61")
    plt.plot(x, y[21 * len(x) : 22 * len(x)], label="LDA (m=9), γ = 0.01, K = 10", color="#6b5b95")
    plt.plot(x, y[22 * len(x) : 23 * len(x)], label="LDA (m=9), γ = 0.1, K = 1", color="#feb236")
    plt.plot(x, y[23 * len(x) : 24 * len(x)], label="LDA (m=9), γ = 0.1, K = 10", color="#d64161")
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["No Pre γ=0.0001 K=1", "No Pre γ=0.0001 K=10", "No Pre γ=0.001 K=1", "No Pre γ=0.001 K=10", "No Pre γ=0.01 K=1", "No Pre γ=0.01 K=10", "No Pre γ=0.1 K=1", "No Pre γ=0.1 K=10", "PCA m=9 γ=0.0001 K=1", "PCA m=9 γ=0.0001 K=10", "PCA m=9 γ=0.001 K=1", "PCA m=9 γ=0.001 K=10", "PCA m=9 γ=0.01 K=1", "PCA m=9 γ=0.01 K=10", "PCA m=9 γ=0.1 K=1", "PCA m=9 γ=0.1 K=10", "LDA m=9 γ=0.0001 K=1", "LDA m=9 γ=0.0001 K=10", "LDA m=9 γ=0.001 K=1", "LDA m=9 γ=0.001 K=10", "LDA m=9 γ=0.01 K=1", "LDA m=9 γ=0.01 K=10", "LDA m=9 γ=0.1 K=1", "LDA m=9 γ=0.1 K=10"], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=3)
    plt.xlabel("C")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

def plotGMM(results, number_of_components, error_rate=False):
    # Define the labels for the x-axis
    #x_labels = [str(10 - i) for i in range(len(results))]
    x_labels = ["No Pre", "PCA m=9", "PCA m=8", "PCA m=7", "LDA m=9", "LDA m=8", "LDA m=7"]

    if error_rate:
        y_label = "Error Rate"
    else:
        y_label = "minDCF"

    # Define the width of the bars
    bar_width = 0.15

    # Define the x-coordinates of the bars for each classifier
    x_FC = np.arange(len(x_labels))
    x_NB = [x + bar_width for x in x_FC]
    x_FTC = [x + bar_width for x in x_NB]

    # Extract the results for each classifier
    results_FC = [results[i][0] for i in range(len(results))]
    results_NB = [results[i][1] for i in range(len(results))]
    results_FTC = [results[i][2] for i in range(len(results))]

    # Create the bar plots
    plt.bar(x_FC, results_FC, color='#ff6f61', width=bar_width, label='FC')
    plt.bar(x_NB, results_NB, color='#6b5b95', width=bar_width, label='NB')
    plt.bar(x_FTC, results_FTC, color='#feb236', width=bar_width, label='FTC')

    # Add labels and title
    #plt.xlabel('Number of Principal Components')
    plt.ylabel(y_label)
    plt.title(f'{y_label} for GMM of {number_of_components} components')

    # Add xticks on the middle of the group bars
    plt.xticks([r + bar_width for r in range(len(x_labels))], x_labels)

    # Set grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Major grid lines
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  # Minor grid lines
    plt.minorticks_on()  # Enable minor ticks

    # Create legend
    plt.legend()

    # Show plot
    plt.show()

def plotLR(results, error_rate=False):
    plt.figure()
    # Define the labels for the x-axis
    #x_labels = [str(10 - i) for i in range(len(results))]
    x_labels = ["No Pre", "PCA m=9", "PCA m=8", "PCA m=7", "LDA m=9", "LDA m=8", "LDA m=7"]

    if error_rate:
        y_label = "Error Rate"
    else:
        y_label = "minDCF"

    # Define the width of the bars
    bar_width = 0.15

    # Define the x-coordinates of the bars for each classifier
    x = np.arange(len(x_labels))
    #x_NB = [x + bar_width for x in x_FC]

    # Extract the results for each classifier
    results = [results[i] for i in range(len(results))]
    #results_NB = [results[i][1] for i in range(len(results))]

    # Create the bar plots
    plt.bar(x, results, color='#ff6f61', width=bar_width, label='FC')
    #plt.bar(x_NB, results_NB, color='#6b5b95', width=bar_width, label='NB')

    # Add labels and title
    #plt.xlabel('Number of Principal Components')
    plt.ylabel(y_label)
    plt.title(f'{y_label} for Logistic Regression')

    # Add xticks on the middle of the group bars
    plt.xticks([r + bar_width for r in range(len(x_labels))], x_labels)

    # Set grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Major grid lines
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  # Minor grid lines
    plt.minorticks_on()  # Enable minor ticks

    # Create legend
    plt.legend()

    # Show plot
    plt.show()

def plotQLR(results, error_rate=False):
    plt.figure()
    # Define the labels for the x-axis
    #x_labels = [str(10 - i) for i in range(len(results))]
    x_labels = ["No Pre", "PCA m=9", "LDA m=9"]

    if error_rate:
        y_label = "Error Rate"
    else:
        y_label = "minDCF"

    # Define the width of the bars
    bar_width = 0.15

    # Define the x-coordinates of the bars for each classifier
    x = np.arange(len(x_labels))
    #x_NB = [x + bar_width for x in x_FC]

    # Extract the results for each classifier
    results = [results[i] for i in range(len(results))]
    #results_NB = [results[i][1] for i in range(len(results))]

    # Create the bar plots
    plt.bar(x, results, color='#ff6f61', width=bar_width, label='FC')
    #plt.bar(x_NB, results_NB, color='#6b5b95', width=bar_width, label='NB')

    # Add labels and title
    #plt.xlabel('Number of Principal Components')
    plt.ylabel(y_label)
    plt.title(f'{y_label} for Logistic Regression')

    # Add xticks on the middle of the group bars
    plt.xticks([r + bar_width for r in range(len(x_labels))], x_labels)

    # Set grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Major grid lines
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  # Minor grid lines
    plt.minorticks_on()  # Enable minor ticks

    # Create legend
    plt.legend()

    # Show plot
    plt.show()

def plotSVM(results, error_rate=False):
    plt.figure()
    # Define the labels for the x-axis
    #x_labels = [str(10 - i) for i in range(len(results))]
    x_labels = ["No Pre", "PCA m=9", "PCA m=8", "LDA m=9", "LDA m=8"]

    if error_rate:
        y_label = "Error Rate"
    else:
        y_label = "minDCF"

    # Define the width of the bars
    bar_width = 0.15

    # Define the x-coordinates of the bars for each classifier
    x = np.arange(len(x_labels))
    #x_NB = [x + bar_width for x in x_FC]

    # Extract the results for each classifier
    results = [results[i] for i in range(len(results))]
    #results_NB = [results[i][1] for i in range(len(results))]

    # Create the bar plots
    plt.bar(x, results, color='#ff6f61', width=bar_width, label='FC')
    #plt.bar(x_NB, results_NB, color='#6b5b95', width=bar_width, label='NB')

    # Add labels and title
    #plt.xlabel('Number of Principal Components')
    plt.ylabel(y_label)
    plt.title(f'{y_label} for SVM')

    # Add xticks on the middle of the group bars
    plt.xticks([r + bar_width for r in range(len(x_labels))], x_labels)

    # Set grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Major grid lines
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')  # Minor grid lines
    plt.minorticks_on()  # Enable minor ticks

    # Create legend
    plt.legend()

    # Show plot
    plt.show()
    
def plotBayesError(actDCF, minDCF, effPriorLog, model):
    plt.figure()
    plt.plot(effPriorLog, actDCF, label="Actual DCF", color="r")
    plt.plot(effPriorLog, minDCF, label="Min DCF", color="b", linestyle="--")
    plt.xlim([min(effPriorLog), max(effPriorLog)])
    plt.legend([f"{model} actDCF", f"{model} minDCF"])
    plt.xlabel("Prior log")
    plt.ylabel("DCF")
    plt.title(f"Bayes Error for {model}")
    plt.show()

def plotBayesErrorCalibrated(actDCF1, minDCF, effPriorLog, model, lambda1):
    plt.figure()
    plt.plot(effPriorLog, actDCF1, label=f"Actual DCF - λ={lambda1}", color="r")
    plt.plot(effPriorLog, minDCF, label="Min DCF", color="b", linestyle="--")
    plt.xlim([min(effPriorLog), max(effPriorLog)])
    plt.legend([f"{model} actDCF - λ={lambda1}", f"{model} minDCF"])
    plt.xlabel("Prior log")
    plt.ylabel("DCF")
    plt.title(f"Bayes Error for {model} - Calibrated")
    plt.show()

def plotROC(FPR0, TPR0, FPR1, TPR1, FPR2, TPR2):
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(FPR0, TPR0, linewidth=2, color='r')
    plt.plot(FPR1, TPR1, linewidth=2, color='b')
    plt.plot(FPR2, TPR2, linewidth=2, color='g')
    plt.legend(["MVG Full-Cov", "GMM Full-Con", "RBF SVM"])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

def plotLambdaQLR(x, y):
    plt.figure()
    plt.plot(x, y[0 : len(x)], label="No Preprocessing", color="b")
    plt.plot(x, y[len(x) : 2 * len(x)], label="PCA (m=9)", color="r")
    plt.plot(x, y[6 * len(x) : 7 * len(x)], label="LDA (m=9)", color="k")
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["No Preprocessing", "PCA (m=9)", "LDA (m=9)"])
    plt.xlabel("λ")
    plt.ylabel("minDCF")
    plt.show()