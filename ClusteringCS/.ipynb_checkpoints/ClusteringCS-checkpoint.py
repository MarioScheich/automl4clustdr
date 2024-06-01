import logging
import time

from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Hyperparameter, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, Constant
from sklearn import manifold, cluster
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch, SpectralClustering, DBSCAN, \
    MeanShift, estimate_bandwidth, AffinityPropagation
from typing import Union, Type, List, Dict

from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.mixture import GaussianMixture
import numpy as np


def execute_algorithm_from_config(X: np.array, config: Configuration, run_complete_algorithm=False):
    algo_instance = ALGORITHMS_MAP[config["algorithm"]]
    labels = algo_instance.execute_config(X, config, run_complete_algorithm)
    return labels


class ClusteringAlgorithm:

    def __init__(self, name: str,
                 parameters: [Hyperparameter],
                 algorithm_class,
                 additional_kwargs=None):

        if additional_kwargs is None:
            additional_kwargs = {}

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.parameters = parameters
        self.name = name
        self.algorithm_class = algorithm_class
        self.additional_kwargs = additional_kwargs

    def execute_config(self, X, configuration: Configuration, run_complete_algorithm=False):
        config_dict = configuration.get_dictionary().copy()

        self.logger.info(f"Executing configuration: {configuration}")

        if run_complete_algorithm:
            additional_algorithm_args = {}
        else:
            additional_algorithm_args = self.additional_kwargs

        self.logger.info(f"Additional kwargs for the algorithm are: {additional_algorithm_args}")

        t0 = time.time()
        if "dim_reduction" in config_dict:
            dim_reduction = config_dict.pop("dim_reduction")

            if dim_reduction != NONE_DR:
                n_components = config_dict.pop("n_components")
                dr_algo = DR_MAP[dim_reduction]
                X = dr_algo(n_components).fit_transform(X)

        algorithm_name = config_dict.pop("algorithm")

        if algorithm_name == MEAN_SHIFT_ALGORITHM:
            if "quantile" in configuration:
                bandwith = estimate_bandwidth(X, config_dict["quantile"])
                labels = self.algorithm_class(bandwidth=bandwith, **additional_algorithm_args).fit_predict(X)
            else:
                labels = [-1 for x in X]
                self.logger.warning(f"Something got wrong with the configuration {configuration}")
                self.logger.warning("Setting labels of all data points to -1")
        else:
            if algorithm_name == GMM_ALGORITHM:
                n_clusters = config_dict.pop("n_clusters")
                labels = self.algorithm_class(n_components=n_clusters, **config_dict,
                                              **additional_algorithm_args).fit_predict(X)
            else:
                algorithm_instance = self.algorithm_class(**config_dict, **additional_algorithm_args)
                labels = algorithm_instance.fit_predict(X)

        algo_time = time.time() - t0
        self.logger.info(f"Configuration executed: {configuration}")
        self.logger.info(f"Execution took {algo_time} seconds")
        return labels


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


MAX_ITERATIONS = 100

PCA_DR = "PCA"
n_components = (2, 100)
FAST_ICA = "FastICA"
FACTOR_ANALYSIS = "FactorAnalysis"
TSNE = "TSNE"
NONE_DR = "None"
dim_reductions = [  # TSNE,
    PCA_DR, FAST_ICA, FACTOR_ANALYSIS, NONE_DR]

DR_MAP = {PCA_DR: PCA, FAST_ICA: FastICA, FACTOR_ANALYSIS: FactorAnalysis,
          # TSNE: manifold.TSNE
          }

KMEANS_ALGORITHM = "KMeans"
GMM_ALGORITHM = "GMM"
# KMEDOIDS = "KMedoids"
MINI_BATCH_KMEANS = "MBKMeans"
WARD_ALGORITHM = "ward"
DBSCAN_ALGORITHM = "dbscan"
BIRCH_ALGORITHM = "birch"
SPECTRAL_ALGORITHM = "spectral"
MEAN_SHIFT_ALGORITHM = "means_shift"
AFFINITY_PROPAGATION_ALGORITHM = "affinity_propagation"

algorithms = [KMEANS_ALGORITHM, GMM_ALGORITHM,
              # KMEDOIDS,
              MINI_BATCH_KMEANS, WARD_ALGORITHM, DBSCAN_ALGORITHM, BIRCH_ALGORITHM,
              SPECTRAL_ALGORITHM, MEAN_SHIFT_ALGORITHM, AFFINITY_PROPAGATION_ALGORITHM]

#TO-DO: adjust setting according to subset size when using hyperband approach
#for big datasets:
#k_range = (2, 200)
#min_samples_range = (2, 200)
#for small datasets:
#k_range = (2, 50)
#min_samples_range = (2, 50)
k_range = (2, 50)
min_samples_range = (2, 50)
eps_range = (0.1, 1.0)
quantile_range = (0.1, 1.0)
damping_range = (0.5, 1.0)

n_clusters_algorithms = [KMEANS_ALGORITHM, MINI_BATCH_KMEANS, GMM_ALGORITHM, SPECTRAL_ALGORITHM, BIRCH_ALGORITHM,
                         WARD_ALGORITHM]

eps_hyperparameter = UniformFloatHyperparameter("eps", lower=eps_range[0],
                                                upper=eps_range[1])

quantile_hyperparameter = UniformFloatHyperparameter("quantile", lower=quantile_range[0],
                                                     upper=quantile_range[1])

damping_hyperparameter = UniformFloatHyperparameter("damping", lower=damping_range[0],
                                                    upper=damping_range[1])

min_samples_hyperparameter = UniformIntegerHyperparameter("min_samples", lower=min_samples_range[0],
                                                          upper=min_samples_range[1])

n_clusters_hyperparameter = UniformIntegerHyperparameter("n_clusters", lower=k_range[0],
                                                         upper=k_range[1], default_value=k_range[0])

n_components_hyperparameter = UniformIntegerHyperparameter("n_components", lower=k_range[0],
                                                           upper=k_range[1], default_value=k_range[0])

ALL_PARAMETERS = [eps_hyperparameter, min_samples_hyperparameter, quantile_hyperparameter,
                  damping_hyperparameter]

################ Definition of algorithms and their hyperparameters #################################################
### todo: maybe adjust additional kwargs
ALGORITHMS_MAP = {KMEANS_ALGORITHM: ClusteringAlgorithm(name=KMEANS_ALGORITHM, algorithm_class=KMeans,
                                                        parameters=[n_clusters_hyperparameter],
                                                        additional_kwargs={"n_init": 1, "max_iter": MAX_ITERATIONS}),
                  GMM_ALGORITHM: ClusteringAlgorithm(name=GMM_ALGORITHM, algorithm_class=GaussianMixture,
                                                     parameters=[n_clusters_hyperparameter],
                                                     additional_kwargs={"n_init": 1, "max_iter": MAX_ITERATIONS}),
                  MINI_BATCH_KMEANS: ClusteringAlgorithm(name=MINI_BATCH_KMEANS, algorithm_class=MiniBatchKMeans,
                                                         parameters=[n_clusters_hyperparameter],
                                                         additional_kwargs={"n_init": 1, "max_iter": MAX_ITERATIONS}),
                  WARD_ALGORITHM: ClusteringAlgorithm(name=WARD_ALGORITHM, algorithm_class=AgglomerativeClustering,
                                                      parameters=[n_clusters_hyperparameter]),
                  BIRCH_ALGORITHM: ClusteringAlgorithm(name=BIRCH_ALGORITHM, algorithm_class=Birch,
                                                       parameters=[n_clusters_hyperparameter]),
                  SPECTRAL_ALGORITHM: ClusteringAlgorithm(name=SPECTRAL_ALGORITHM, algorithm_class=SpectralClustering,
                                                          parameters=[n_clusters_hyperparameter],
                                                          additional_kwargs={"n_init": 1,
                                                                             # "n_jobs": -1
                                                                             }),
                  DBSCAN_ALGORITHM: ClusteringAlgorithm(name=DBSCAN_ALGORITHM, algorithm_class=DBSCAN,
                                                        parameters=[eps_hyperparameter, min_samples_hyperparameter],
                                                        # additional_kwargs={"n_jobs": -1}
                                                        ),
                  MEAN_SHIFT_ALGORITHM: ClusteringAlgorithm(name=MEAN_SHIFT_ALGORITHM, algorithm_class=MeanShift,
                                                            parameters=[quantile_hyperparameter],
                                                            additional_kwargs={"max_iter": MAX_ITERATIONS,
                                                                               # "n_jobs": -1
                                                                               }),
                  AFFINITY_PROPAGATION_ALGORITHM: ClusteringAlgorithm(name=AFFINITY_PROPAGATION_ALGORITHM,
                                                                      algorithm_class=AffinityPropagation,
                                                                      parameters=[damping_hyperparameter],
                                                                      additional_kwargs={"max_iter": MAX_ITERATIONS}),
                  }


def build_paramter_space_per_algorithm(algorithms=algorithms):
    """

    :param algorithm_map:
    :return: configuration space for each algorithm in a map, where key is the name of the algorithm
    """
    cs_per_algo = {}
    for algo_name in algorithms:
        algo = ALGORITHMS_MAP[algo_name]
        cs = ConfigurationSpace()

        algorithm_hyperparameter = CategoricalHyperparameter("algorithm", choices=[algo_name],
                                                             default_value=algo_name
                                                             )
        cs.add_hyperparameter(algorithm_hyperparameter)

        params = algo.parameters
        for param in params:
            cs.add_hyperparameter(param)
            cs.add_condition(InCondition(param, algorithm_hyperparameter, [algo_name]))
        cs_per_algo[algo_name] = cs
    return cs_per_algo


def build_kmeans_space(n_samples=None, n_features=None):
    return build_config_space(clustering_algorithms=[KMEANS_ALGORITHM],
                              dim_reductions=[], partitional=True)
    

def build_kmeans_ward_space(n_samples=None, n_features=None):
    return build_config_space(clustering_algorithms=[KMEANS_ALGORITHM, WARD_ALGORITHM],
                              dim_reductions=[], partitional=True)


def build_partitional_config_space(n_samples=None, n_features=None):
    return build_config_space(clustering_algorithms=[KMEANS_ALGORITHM, MINI_BATCH_KMEANS, GMM_ALGORITHM],
                              dim_reductions=[], partitional=True)


def build_partitional_dim_reduction_space(n_samples=1000, n_features=10):
    return build_config_space(clustering_algorithms=[KMEANS_ALGORITHM, MINI_BATCH_KMEANS, GMM_ALGORITHM],
                              dim_reductions=dim_reductions, partitional=True, n_samples=n_samples,
                              n_features=n_features)


def build_all_algos_space(n_samples=None, n_features=None):
    return build_config_space(clustering_algorithms=algorithms, dim_reductions=[], partitional=False)


def build_all_algos_dim_reduction_space(n_samples=1000, n_features=10):
    return build_config_space(clustering_algorithms=algorithms, dim_reductions=dim_reductions, partitional=False,
                              n_samples=n_samples, n_features=n_features)


def build_config_space(clustering_algorithms=algorithms, n_samples=None, n_features=None, dim_reductions=dim_reductions,
                       partitional=False):
    """
    Builds configuration space of algorithms and their hyperparameters.
    Uses the algorithms from the algorithms list of clustering_algorithms is not provided.
    The configuration space is builded, such that the hyperparameters of the algorithms are conditional on the
     algorithm.
    :param clustering_algorithms: clustering algorithms to use. Should be a list of Clustering algorithm names,
        see algorithms.
    :param dim_reductions: Algorithms for Dimension Reduction to use.
    :param n_features: number of features of a dataset. Is only used if there is at least one dimension reduction
        algorithm.
    :param n_samples: Number of samples of the dataset. Is only used if there is at least one dimension reduction
        algorithm.
    :param partitional: True or False. If true, then only three partitional clustering algorithms are used.
    :return: ConfigurationSpace object that describes the config space with the algorithms and their default parameters.
    """
    cs = ConfigurationSpace()

    if dim_reductions:
        dim_reduction_hp = CategoricalHyperparameter("dim_reduction", choices=dim_reductions)
        print(n_samples)
        print(n_features)
        if n_samples and n_features:
            n_components = (1, min(n_samples, n_features))
        n_components_hp = UniformIntegerHyperparameter("n_components", lower=n_components[0], upper=n_components[1])

        cs.add_hyperparameter(dim_reduction_hp)
        cs.add_hyperparameter(n_components_hp)
        cs.add_condition(InCondition(n_components_hp, dim_reduction_hp, [dim_red for dim_red in dim_reductions
                                                                         if dim_red != NONE_DR]))

    if clustering_algorithms:
        algorithm_hyperparameter = CategoricalHyperparameter("algorithm", choices=clustering_algorithms,
                                                             default_value=clustering_algorithms[0]
                                                             )
        cs.add_hyperparameter(algorithm_hyperparameter)

        for algorithm in clustering_algorithms:
            params = ALGORITHMS_MAP[algorithm].parameters
            for param in params:
                if param.name == n_clusters_hyperparameter.name:
                    # todo: we could add here n_custers for each algorithm separately.
                    #  But we want one n_clusters parameter for all algorithms
                    continue
                cs.add_hyperparameter(param)
                cs.add_condition(InCondition(param, algorithm_hyperparameter, [algorithm]))

        # add n_clusters hyperparameter separately
        cs.add_hyperparameter(n_clusters_hyperparameter, )
        if not partitional:
            cs.add_condition(InCondition(n_clusters_hyperparameter, algorithm_hyperparameter,
                                         # add n_clusters algorithms that are also contained in the parameter list clustering_algorithms
                                         [algorithm for algorithm in clustering_algorithms
                                          if algorithm in n_clusters_algorithms]))
        if partitional:
            cs.add_condition(InCondition(n_clusters_hyperparameter, algorithm_hyperparameter,
                                         clustering_algorithms))
        return cs


PARTITIONAL_SPACE = "partitional"
# PARTITIONAL_DR_SPACE = "partitional+dr"
KMEANS_SPACE = "kmeans"
ALL_ALGOS_SPACE = "all_algos"
KMEANS_WARD_SPACE = "pipeline_space"

CONFIG_SPACE_MAPPING = {PARTITIONAL_SPACE: build_partitional_config_space(), KMEANS_SPACE: build_kmeans_space(),
                        ALL_ALGOS_SPACE: build_all_algos_space(), KMEANS_WARD_SPACE: build_kmeans_ward_space()}
