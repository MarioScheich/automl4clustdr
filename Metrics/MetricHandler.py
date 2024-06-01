from ConfigSpace.configuration_space import Configuration
from sklearn import metrics
import time
import logging
import math
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AffinityPropagation
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler
from hdbscan import validity_index as dbcv_score
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from numpy import mean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ClusteringCS import ClusteringCS
from Metrics import DunnIndex, CogginsJain, COP_Index

"""
Responsible for everything related to Metrics.
It contains all metrics, the metricresult class, the collection of metrics that are used, the metric class itself 
and the MetricEvaluator.
"""


class MetricResult:
    """
        Class that describes the information that is saved for each metric after calculating the metric result for a
        given kmeans result. Is used to represent the result of the MetricEvaluator.run_metrics() method.
    """

    def __init__(self, execution_time, score, name, metric_type):
        self.execution_time = execution_time
        self.score = score
        self.name = name
        self.metric_type = metric_type


class MetricType:
    EXTERNAL = "External"
    INTERNAL = "Internal"
    COMPLEXITY_MEASURE = "CM"


class MetricObjective:
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Metric:
    """
        Basic entity that describes a metric. For each metric there is one instance of this class which will be saved in
        the MetricCollection class.
        This class is also responsible for evaluating the metric score in a generic way.
    """

    def __init__(self, name, score_function, metric_type, metric_objective=MetricObjective.MAXIMIZE, sample_size=None):
        self.name = name
        self.score_function = score_function
        self.metric_type = metric_type
        self.sample_size = sample_size
        self.metric_objective = metric_objective

    def get_abbrev(self):
        return MetricCollection.METRIC_ABBREVIATIONS[self.name]

    def score_metric(self, data, labels=None, true_labels=None, y_train=None):

        """
        Calculates the score of a metric for a given dataset and the corresponding class labels. If the metric is an
        external metric, also the true_labels have to be passed to calculate the metric. :param data: the raw dataset
        without labels
        :param y_train: target values for given data, needed for complexity measures
        :param data:
        :param labels: the labels that were calculated for example by kmeans
        :param true_labels: the gold standard labels of the dataset (is needed for external metrics)
        :return: the result of the metric calculation, which should be a float. It is the negative value of a metric if
        the metric should be optimized (since we want to minimize the value)
        """
        if labels is None and true_labels is None:
            logging.error("either labels or true_labels has to be set to get metric score")
            raise Exception("either labels or true_labels has to be set to get metric score")

        # set default score to highest possible score --> Infinity will throw exception later on
        score = 2147483647

        logging.info("Start scoring metric {}".format(self.name))

        """ TODO: Maybe we should only remove noise for the ARI?! --> DO not do it for DBCV
        if -1 in labels:
            logging.info("-1 in labels detected, this is an indication that the algorithm can handle noise")
            not_noise_indices = [i for i, x in enumerate(labels) if x != -1]
            data =  np.take(data, not_noise_indices, axis=0)
            labels = np.take(labels, not_noise_indices, axis=0)
            true_labels = np.take(true_labels, not_noise_indices, axis=0)
        """
        if (len(np.unique(labels)) == 1 or len(np.unique(labels)) == 0) and self.metric_type == MetricType.INTERNAL:
            logging.warning("only 1 label - return default value")
            return score

        if -2 in labels and self.metric_type == MetricType.INTERNAL:
            logging.warning("-2 in labels detected, this is an indication for cutoff. Returning default value")
            return score

        # if internal just calculate score by data and labels
        if self.metric_type == MetricType.INTERNAL:
            score = self.score_function(data, labels)

        # if external metric then we need the "ground truth" instead of the data
        elif self.metric_type == MetricType.EXTERNAL:
            score = self.score_function(true_labels, labels)
        # if complexity measure we need target values from the data
        elif self.metric_type == MetricType.COMPLEXITY_MEASURE:
            score = self.score_function(data, labels, y_train)
        else:
            logging.error("There was an unknown metric type which couldn't be calculated. The metric is " + self.name)

        if math.isnan(score):
            logging.info("Scored metric {} and value is NAN. Returning 2147483647 as value".format(self.name))
            return 2147483647

        if self.metric_objective == MetricObjective.MAXIMIZE:
            score = -1 * score

        logging.info("Scored metric {} and value is {}".format(self.name, score))
        return score


def dunn_score(X, labels):
    #### Taken from https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py####
    return DunnIndex.dunn_fast(X, labels)


def fisher_disc_score(data, labels, y_train):
    return fdr_score(data, labels, y_train)


def error_rate_knn_score(data, labels, y_train):
    return error_rate_knn(data, labels, y_train)


def pca_to_raw_score(data, labels, y_train):
    return pca_to_raw_ratio(data, labels, y_train)


def density_based_score(X, labels):
    try:
        return dbcv_score(X, labels)
    except ValueError as ve:
        logging.error(f"Error occured: {ve}")
        return -1.0


def error_rate_knn(X, labels):
    # perform cross validation using leave-one-out method
    model = KNeighborsClassifier()
    cv = LeaveOneOut()
    scores = cross_val_score(model, X, labels, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    accuracy_mean = mean(scores)
    # get error rate
    error_rate = 1-accuracy_mean
    return 1-error_rate


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






class MLPMetric(Metric):

    def __init__(self, mlp_model):
        super().__init__(name="MLPMetric", score_function=None, metric_type=MetricType.INTERNAL,
                         metric_objective=MetricObjective.MINIMIZE)
        self.mlp_model = mlp_model

    def score_metric(self, data, labels=None, true_labels=None):
        print("scoring mlp metric")
        metric_scores = []

        # calculate all internal metrics
        for metric in MetricCollection.internal_metrics:
            metric_score = metric.score_metric(data, labels)
            print(f"metric score for {metric.name} is: {metric_score}")
            if math.isnan(metric_score) or math.isinf(metric_score):
                metric_score = 2147483647
            metric_scores.append(metric_score)
        metric_scores = np.array(metric_scores).reshape(1, -1)

        # predict ARI score based on the internal metric scores
        ari_score = self.mlp_model.predict(metric_scores)

        return ari_score[0]


class MetricCollection:
    """
        Contains all metrics that are used for the experiments. The metrics can be get by either calling all_metrics or
        using the get_all_metrics_sorted method.
    """

    # SILHOUETTE_SAMPLE = "Sampled Silhouette"
    # SILHOUETTE_SAMPLE_10 = Metric(SILHOUETTE_SAMPLE, metrics.silhouette_score, MetricType.INTERNAL)

    # internal scores to maximize
    SILHOUETTE = Metric("Silhouette", metrics.silhouette_score, MetricType.INTERNAL)
    CALINSKI_HARABASZ = Metric("Calinski-Harabasz", metrics.calinski_harabasz_score, MetricType.INTERNAL)
    DUNN_INDEX = Metric("Dunn Index", dunn_score, MetricType.INTERNAL)
    DENSITY_BASED_VALIDATION = Metric("DBCV", density_based_score, MetricType.INTERNAL)
    COGGINS_JAIN_INDEX = Metric("Coggins Jain Index", CogginsJain.coggins_jain_score, MetricType.INTERNAL)

    # complexity measures
    FISHER_DISCRIMINANT_RATIO = Metric("Fisher Disc Ratio", fisher_disc_score, MetricType.COMPLEXITY_MEASURE)
    PCA_TO_RAW_DIMENSIONS_RATIO = Metric("PCA Reduced to Raw Dimensions Ratio", pca_to_raw_score,
                                         MetricType.COMPLEXITY_MEASURE)
    ERROR_RATE_KNN = Metric("Error Rate KNN", error_rate_knn_score, MetricType.COMPLEXITY_MEASURE)

    # internal scores to maximize
    DAVIES_BOULDIN = Metric("Davies-Bouldin", metrics.davies_bouldin_score, MetricType.INTERNAL,
                            MetricObjective.MINIMIZE)
    COP_SCORE = Metric("COP", COP_Index.cop_score, MetricType.INTERNAL, MetricObjective.MINIMIZE)

    # external scores
    ADJUSTED_RAND = Metric("Adjusted Rand", metrics.adjusted_rand_score, MetricType.EXTERNAL)
    ADJUSTED_MUTUAL = Metric("Adjusted Mutual", metrics.adjusted_mutual_info_score, MetricType.EXTERNAL)
    HOMOGENEITY = Metric("Homogeneity", metrics.homogeneity_score, MetricType.EXTERNAL)
    V_MEASURE = Metric("V-measure", metrics.v_measure_score, MetricType.EXTERNAL)
    COMPLETENESS_SCORE = Metric("Completeness", metrics.completeness_score, MetricType.EXTERNAL)
    FOWLKES_MALLOWS = Metric("Folkes-Mallows", metrics.fowlkes_mallows_score, MetricType.EXTERNAL)

    # abbreviations are useful for, e.g., plots
    METRIC_ABBREVIATIONS = {
        SILHOUETTE.name: "SIL",
        CALINSKI_HARABASZ.name: "CH",
        DAVIES_BOULDIN.name: "DBI",
        DUNN_INDEX.name: "DI",
        DENSITY_BASED_VALIDATION.name: "DBCV",
        COP_SCORE.name: "COP",
        COGGINS_JAIN_INDEX.name: "CJI",
        ADJUSTED_RAND.name: "ARI",
        ADJUSTED_MUTUAL.name: "AMI",
        HOMOGENEITY.name: "HG",
        V_MEASURE.name: "VM",
        COMPLETENESS_SCORE.name: "CS",
        FOWLKES_MALLOWS.name: "FM",
        "MLPMetric": "MLP",
        FISHER_DISCRIMINANT_RATIO.name: "FDR",
        PCA_TO_RAW_DIMENSIONS_RATIO.name: "PCARR",
        ERROR_RATE_KNN.name: "EKNN"
    }
    internal_metrics = [CALINSKI_HARABASZ,
                        DAVIES_BOULDIN,
                        # SILHOUETTE_SAMPLE_10,
                        SILHOUETTE,
                        # added scores
                        DENSITY_BASED_VALIDATION,
                        DUNN_INDEX,
                        COGGINS_JAIN_INDEX,
                        COP_SCORE
                        ]
    external_metrics = [ADJUSTED_MUTUAL, ADJUSTED_RAND,
                        COMPLETENESS_SCORE,
                        FOWLKES_MALLOWS,
                        HOMOGENEITY, V_MEASURE]
    complexity_measures = [FISHER_DISCRIMINANT_RATIO,
                           PCA_TO_RAW_DIMENSIONS_RATIO,
                           ERROR_RATE_KNN
                           ]
    all_metrics = external_metrics + internal_metrics + complexity_measures
    experiment_metrics = [CALINSKI_HARABASZ, DAVIES_BOULDIN,
                          # SILHOUETTE,
                          ADJUSTED_MUTUAL]

    @staticmethod
    def get_metric_by_abbrev(metric_abbrev):
        print(metric_abbrev)
        for metric in MetricCollection.all_metrics:
            if MetricCollection.METRIC_ABBREVIATIONS[metric.name] == metric_abbrev:
                return metric

    @staticmethod
    def get_all_metrics_sorted_by_name():
        """
        Returns all metrics in sorted order. This is important, if e.g. calculations were done and you want to map
        value to their corresponding name.
        :return:
        """
        MetricCollection.all_metrics.sort(key=lambda x: x.name)
        return MetricCollection.all_metrics

    @staticmethod
    def get_sorted_abbreviations_by_type():
        return [MetricCollection.METRIC_ABBREVIATIONS[metric.name] for metric
                in MetricCollection.all_metrics]

    @staticmethod
    def get_sorted_abbreviations_internal_by_type():
        return [MetricCollection.METRIC_ABBREVIATIONS[metric.name] for metric
                in MetricCollection.internal_metrics]

    @staticmethod
    def get_abrev_for_metric(metric_name):
        return MetricCollection.METRIC_ABBREVIATIONS[metric_name]


class MetricEvaluator:

    @staticmethod
    def run_metrics(data, true_labels, labels):
        """
        :param data: dataset that is the raw dataset and without labels
        :param true_labels: the labels of the ground truth
        :param labels: predicted labels that were found by the clustering algorithm
        :return: List of :py:class:`MetricResult`, one for each Metric that was used.
        """
        logging.info("start calculating metrics")
        result = []

        for metric in MetricCollection.all_metrics:
            metric_name = metric.name
            logging.info("start calculation for " + metric_name)
            metric_execution_start = time.time()
            score = metric.score_metric(data, labels=labels, true_labels=true_labels)
            metric_execution_time = time.time() - metric_execution_start
            metric_result = MetricResult(name=metric.name, score=score, execution_time=metric_execution_time,
                                         metric_type=metric.metric_type)
            logging.info("Finished {} with score {} and execution time {}"
                         .format(metric_name, score, metric_execution_time))
            result.append(metric_result)
        return result


if __name__ == "__main__":
    pass
