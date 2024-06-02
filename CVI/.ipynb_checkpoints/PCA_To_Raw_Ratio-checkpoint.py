from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_to_raw_ratio(X, labels):
    # get raw dimensions
    regular_dimensions = X.shape[1]
    # normalize data
    x = StandardScaler().fit_transform(X)
    # pca to keep 95% of original data variations
    pca = PCA(0.95)
    reduced_dataset = pca.fit_transform(x)
    # get reduced dimensions
    reduced_dimensions = reduced_dataset.shape[1]
    # calculate ratio
    ratio = reduced_dimensions/regular_dimensions
    return 1-ratio
