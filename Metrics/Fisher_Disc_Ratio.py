import numpy as np
from numpy.linalg import norm
from numpy import mean

def fdr_score(X, labels):
    elements = X.shape[0]
    features = X.shape[1]
    n_labels = np.unique(labels)
    # calculate ratio for each feature
    for i in range(features):
        element_list = []
        # get feature values and mean
        for h in range(elements):
            element_list.append(X[h][i])
        mean_feature = mean(element_list)
        sum_1 = 0
        for j in n_labels:
            # get class labels
            class_positions = np.where(labels == j)
            n_class_samples = len(class_positions)+1
            class_list = []
            # get class values and mean
            for h in class_positions:
                class_list.append(X[h][i])
            mean_class = mean(class_list)
            # calculate sums
            sum_1 = sum_1 + (n_class_samples*norm(mean_feature-mean_class))
            for h in class_list:
                sum_2 = sum_2 + norm(h-mean_class)
        pre_ratio = pre_ratio + sum_1/sum_2
    ratio = pre_ratio/elements
    return ratio
