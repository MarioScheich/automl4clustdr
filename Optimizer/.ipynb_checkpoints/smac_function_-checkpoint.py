import logging
import multiprocessing
import time

from sklearn.cluster import estimate_bandwidth, MeanShift, KMeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from ClusteringCS import ClusteringCS
from ClusteringCS.ClusteringCS import MEAN_SHIFT_ALGORITHM, GMM_ALGORITHM
from Metrics.MetricHandler import MetricCollection, MetricType
from DataReduction import DataReductionHandler as dr
import numpy as np


def smac_function(config, optimizer_instance, budget=0, **kwargs): #budget_indices, budget=0, **kwargs):
    X = optimizer_instance.dataset
    metric = optimizer_instance.metric
    true_labels = optimizer_instance.true_labels
    algorithm_name = config["algorithm"]
    if budget and budget > 0:
        print("budget:"+ str(budget))
        reduction = list(kwargs.values())
        data_reduction = reduction[0]
        data_reduction_param = reduction[1]
        max_budget = reduction[2]
        if data_reduction == "UniformRandomSampling" or data_reduction == "LightWeightCoreset":
            # round size to nearest integer in order to avoid errors
            data_reduction_param['sample_size'] = round(len(X)*(budget/max_budget))
            # set minimum sample_size to 50 in order to avoid more clusters than samples
            # TODO: should be tied to maximum of possible clusters for optimiziation
            if data_reduction_param['sample_size'] < 50:
                data_reduction_param['sample_size'] = 50 
        else:
            # round feature number to nearest integer in order to avoid errors
            data_reduction_param['feature_number'] = round(X.shape[1]*(budget/max_budget))
            # avoid low percentages leading to zero features chosen
            if data_reduction_param['feature_number'] <1:
                data_reduction_param['feature_number'] = 1
        #sampled_X = dr.data_reduction_main_method(X, data_reduction, data_reduction_param)
        sampled_X = dr.DataReductionHandler(X, data_reduction, data_reduction_param).data_reduction_main_method()
        #choices = budget_indices[int(budget)]
        #sampled_X = X[choices]
    else:
        sampled_X = X
    t0 = time.time()

    clust_algo_instance = ClusteringCS.ALGORITHMS_MAP[algorithm_name]

    # todo: we need to scale the data!
    y = clust_algo_instance.execute_config(sampled_X, config)
    algo_runtime = time.time() - t0

    metric_start = time.time()

    # Scoring metric, true_labels are none for internal metric
    score = metric.score_metric(sampled_X, labels=y, true_labels=true_labels)
    print(f"score: {score}")
    metric_runtime = time.time() - metric_start
    add_info = {"algo_time": algo_runtime, "metric_time": metric_runtime, "labels": y.tolist()}

    if optimizer_instance.metric.metric_type == MetricType.INTERNAL:
        # if we are using an internal metric, this is the online phase and we do not want to calculate all metrics
        # additionally
        # todo: we could use ARI here?
        return score#, add_info

    int_metrics = MetricCollection.internal_metrics
    for int_metric in int_metrics:
      #  int_metric_time = time.time()
        int_metric_score = int_metric.score_metric(X, labels=y)
     #   int_metric_time = time.time() - int_metric_time
        add_info[int_metric.get_abbrev()] = int_metric_score
    #    add_info[int_metric.get_abbrev()] = int_metric_time
    return score#, add_info
