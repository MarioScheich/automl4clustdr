import logging
import os
import time
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn import datasets
from sklearn.cluster import KMeans, MeanShift
from typing import Type

from sklearn.datasets import make_blobs
from sklearn.neighbors.kd_tree import KDTree

from ClusteringCS import ClusteringCS
from ClusteringCS.ClusteringCS import build_partitional_config_space, \
    build_all_algos_dim_reduction_space, build_partitional_dim_reduction_space, build_all_algos_space
from Metrics import MetricHandler
from Metrics.MetricHandler import MetricCollection
from itertools import cycle, islice

from Optimizer.OptimizerSMAC import SMACOptimizer, HyperbandOptimizer, BOHBOptimizer, AbstractOptimizer
from Utils.FileUtil import FileExporter


class DataType:
    # todo: Maybe split up to distribution? For moons, spirals etc.
    SYNTHETIC_GAUSSIAN_TYPE = "synthetic_gaussian"
    SYNTHETIC_SHAPE_TYPE = "synthetic_shapes"
    REAL_TYPE = "real"


def get_dataset_input_path(data_type):
    return f"/volume/data/{data_type}/"


def result_already_exists(file_path):
    trajectory_path = os.path.join(file_path, AllAlgosExperiment.TRAJECTORY_FILENAME)
    history_path = os.path.join(file_path, AllAlgosExperiment.HISTORY_FILENAME)
    return os.path.isfile(trajectory_path) and os.path.isfile(history_path)


class AbstractExperiment(ABC):
    # Initializations
    WARMSTART_INIT = "warmstart"
    COLDSTART_INIT = "coldstart"

    # Phases
    ONLINE_PHASE = "online"
    OFFLINE_PHASE = "offline"

    TRAJECTORY_FILENAME = "trajectory.csv"
    HISTORY_FILENAME = "history.csv"

    def __init__(self, dataset_type=DataType.SYNTHETIC_GAUSSIAN_TYPE, optimizers=None, metrics=None,
                 n_loops=100, initializations=None, n_similar_datasets=10, configspaces=None, hosts=None,
                 n_repetitions=5, phase=ONLINE_PHASE, skip_existing=True, n_jobs=1, experiment_name="ex"):

        if initializations is None:
            initializations = [AllAlgosExperiment.COLDSTART_INIT]
        if hosts is None:
            hosts = ["localhost"]
        if optimizers is None:
            optimizers = [SMACOptimizer, HyperbandOptimizer, BOHBOptimizer]
        if metrics is None:
            metrics = [MetricCollection.CALINSKI_HARABASZ]
        if configspaces is None:
            configspaces = AllAlgosExperiment.config_spaces

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.phase = phase
        self.dataset_type = dataset_type

        self.optimizers = optimizers
        self.metrics = metrics
        self.n_loops = n_loops
        self.initializations = initializations
        self.n_similar_datasets = n_similar_datasets
        self.config_spaces = configspaces
        self.hosts = hosts
        self.n_repetitions = n_repetitions
        self.experiment_name = experiment_name
        self.output_dir_format = "/volume/out/{dataset_type}/{experiment_name}/{cs}/{phase}/{init}/{optimizer}/{metric}" \
                                 "/{repetition}/{file_name}"
        self.ex_runs = []
        self.skip_existing = skip_existing
        self.n_jobs = n_jobs

    def run_experiments(self):
        file_names = self._list_datasets()
        for file_name in file_names:
            for cs_name in self.config_spaces.keys():
                cs = self.config_spaces[cs_name]
                for optimizer in self.optimizers:
                    for metric in self.metrics:
                        for init in self.initializations:
                            for repetition in range(self.n_repetitions):

                                # names of optimizer, metric and dataset
                                optimizer_name = optimizer.get_abbrev()
                                metric_name = metric.get_abbrev()
                                dataset_name = file_name.replace(".csv", "")

                                # get output directory and check if result already exists
                                out_directory = AbstractExperiment.get_output_dir(cs_name, dataset_name, optimizer_name,
                                                                                  metric_name, repetition,
                                                                                  self.dataset_type,
                                                                                  self.phase, init,
                                                                                  self.experiment_name)
                                if self.skip_existing and result_already_exists(out_directory):
                                    self.logger.info(f"skipping experiment for {out_directory}")
                                    continue
                                self.run_experiment_loop(optimizer, metric, file_name, cs, out_directory, init)

    @abstractmethod
    def run_experiment_loop(self, optimizer, metric, file_name, cs, out_directory, init):
        pass

    def _calculate_external_scores(self, opt_instance):
        ex_metrics = [MetricCollection.ADJUSTED_RAND, MetricCollection.ADJUSTED_MUTUAL]
        return self._calculate_metric_scores(opt_instance, ex_metrics)

    def _calculate_internal_scores(self, opt_instance):
        internal_metrics = MetricCollection.internal_metrics
        return self._calculate_metric_scores(opt_instance, internal_metrics)

    def _calculate_metric_scores(self, opt_instance, metrics):
        X = opt_instance.dataset
        y_true = opt_instance.true_labels
        traj_dict_list = opt_instance.get_trajectory()
        self.logger.info(f"trajectory is: {traj_dict_list}")
        # iterate trhough list of incumbent dicts
        for dic in traj_dict_list:
            # get incumbent, execute it and score AMI/ARI score
            incumbent = dic["incumbent"]
            self.logger.info(f"incumbent is: {incumbent}")
            y_pred = ClusteringCS.execute_algorithm_from_config(X, incumbent, True)

            # score the external metrics based on the predicted labels
            for ex_metric in metrics:
                ex_score = ex_metric.score_metric(data=X, labels=y_pred, true_labels=y_true)
                dic[ex_metric.get_abbrev()] = ex_score

        opt_instance.set_trajectory(traj_dict_list)
        return traj_dict_list

    def run_optimizer(self, X: np.array, cs: Type[ConfigurationSpace], optimizer: Type[SMACOptimizer],
                      metric: Type[MetricHandler.Metric], initialization: str, y_true=None):
        opt_instance = optimizer(cs=cs, dataset=X, n_loops=self.n_loops, metric=metric, true_labels=y_true)

        if initialization == AbstractExperiment.WARMSTART_INIT and self.phase == AbstractExperiment.ONLINE_PHASE:
            # todo: check initialization --> warmstart
            init_configs = []

        opt_instance.optimize()
        return opt_instance

    def _list_datasets(self):
        return os.listdir(get_dataset_input_path(self.dataset_type))

    def _load_dataset(self, file_name):
        """
        Imports the data that is specified by filepath. It returns a tuple that has first the dataset as np.array
        without labels and as second entry the labels of the dataset.
        :param filepath: path to csv file of the dataset.
        :return: tuple containing dataset without labels and the true_labels
        """
        path = get_dataset_input_path(self.dataset_type)
        dataset_path = os.path.join(path, file_name)

        self.logger.info("Importing {}".format(dataset_path))
        raw_data = pd.read_csv(dataset_path, header=None)
        self.logger.info("Imported {}, shape is {}".format(dataset_path, raw_data.shape))
        n_features = raw_data.shape[1]
        true_labels = np.array(raw_data[n_features - 1])
        data_without_labels = raw_data[raw_data.columns[:-1]]
        self.logger.info("data (without labels) has shape {}".format(data_without_labels.shape))
        return data_without_labels, true_labels

    @staticmethod
    def get_output_dir(cs_name, dataset_name, optimizer_name, metric_name, repetition,
                       dataset_type=DataType.SYNTHETIC_GAUSSIAN_TYPE, phase=ONLINE_PHASE, init=WARMSTART_INIT,
                       experiment="ex"):
        return f"/volume/out/{experiment}/{dataset_type}/{cs_name}/{phase}/{init}/{optimizer_name}/{metric_name}/{repetition}/" \
               f"{dataset_name}"

    @staticmethod
    def _export_result(opt_instance, out_dir):

        trajectory = opt_instance.get_trajectory()
        trajectory_df = pd.DataFrame(trajectory)

        run_history = opt_instance.get_run_hitory_list_dicts()
        history_df = pd.DataFrame(run_history)

        FileExporter.export_dataframe_to_csv(trajectory_df, out_dir, file_name=AbstractExperiment.TRAJECTORY_FILENAME)
        FileExporter.export_dataframe_to_csv(history_df, out_dir, file_name=AbstractExperiment.HISTORY_FILENAME)


class MetricLearningOfflineExperiment(AbstractExperiment):
    # config space for tuning hyperparameters of each algorithm separately
    config_spaces_dict = ClusteringCS.build_paramter_space_per_algorithm()

    def __init__(self, dataset_type=DataType.SYNTHETIC_GAUSSIAN_TYPE, optimizers=[SMACOptimizer],
                 metrics=[MetricCollection.ADJUSTED_RAND],  # we use external metric for offline phase
                 n_loops=100,
                 initializations=[AbstractExperiment.WARMSTART_INIT], n_similar_datasets=10,
                 configspaces=config_spaces_dict, hosts=None, n_repetitions=5,
                 phase=AbstractExperiment.OFFLINE_PHASE, skip_existing=True, n_jobs=1,
                 experiment_name="metric_learning"):
        super().__init__(dataset_type=dataset_type, optimizers=optimizers, metrics=metrics, n_loops=n_loops,
                         initializations=initializations, n_similar_datasets=n_similar_datasets,
                         configspaces=configspaces, hosts=hosts, n_repetitions=n_repetitions, phase=phase,
                         skip_existing=skip_existing, n_jobs=n_jobs, experiment_name=experiment_name)

    def run_experiment_loop(self, optimizer, metric, file_name, cs, out_directory, init):
        # load the dataset
        X, y_true = self._load_dataset(file_name)

        # instantiate configuration space
        cs_instance = cs

        print(cs_instance)

        # instantiate optimizer and run the optimization
        opt_instance = self.run_optimizer(X, cs_instance, optimizer, metric, init, y_true=y_true)

        self._calculate_internal_scores(opt_instance)
        self._export_result(opt_instance, out_directory)


class AllAlgosExperiment(AbstractExperiment):
    # Initializations
    WARMSTART_INIT = "warmstart"
    COLDSTART_INIT = "coldstart"

    # Phases
    ONLINE_PHASE = "online"
    OFFLINE_PHASE = "offline"

    TRAJECTORY_FILENAME = "trajectory.csv"
    HISTORY_FILENAME = "history.csv"

    # config spaces
    config_spaces = {
        "part": build_partitional_config_space,
        # "part+dr": build_partitional_dim_reduction_space,
        "all": build_all_algos_space,
        # "all+dr": build_all_algos_dim_reduction_space
    }

    def __init__(self, dataset_type=DataType.SYNTHETIC_GAUSSIAN_TYPE, optimizers=None, metrics=None,
                 n_loops=100, initializations=None, n_similar_datasets=10, configspaces=config_spaces, hosts=None,
                 n_repetitions=5, phase=ONLINE_PHASE, skip_existing=True, n_jobs=1, experiment_name="ex"):
        super().__init__(dataset_type=dataset_type, optimizers=optimizers, metrics=metrics, n_loops=n_loops,
                         initializations=initializations, n_similar_datasets=n_similar_datasets,
                         configspaces=configspaces, hosts=hosts, n_repetitions=n_repetitions, phase=phase,
                         skip_existing=skip_existing, n_jobs=n_jobs, experiment_name=experiment_name)

    def run_experiment_loop(self, optimizer, metric, file_name, cs, out_directory, init):
        # load the dataset
        X, y_true = self._load_dataset(file_name)

        # instantiate configuration space
        cs_instance = cs(n_samples=X.shape[0], n_features=X.shape[1])

        # instantiate optimizer and run the optimization
        opt_instance = self.run_optimizer(X, cs_instance, optimizer, metric, init)

        # use the opt instance to calculate the AMI score and also export the results
        self._calculate_external_scores(opt_instance)
        self._export_result(opt_instance, out_directory)


if __name__ == '__main__':
    from pymfe.mfe import MFE
    dataset_sets = []
    label_sets = []
    mfs = []

    our_extraction_time = 0

    for i in range(10):
        X,y = make_blobs(n_samples=100 * (i+1), n_features=i+2, centers=i+5)
        label_sets.append(y)
        dataset_sets.append(X)

        start_our = time.time()
        mfe = MFE()
        mfe.fit(X)
        ft = mfe.extract()
        end_our = time.time() - start_our
        our_extraction_time += end_our
        array = ft[1]
        ft = {x: y for x,y in zip(ft[0], ft[1])}
        mfs.append(np.nan_to_num(array, nan=0))
        print(ft)
    print(f"Ours took overall: {our_extraction_time} seconds")

    kdt = KDTree(np.array(mfs), metric='manhattan')
    X, y = make_blobs(n_samples=100, n_features=10, centers=10)
    mfe = MFE()
    mfe.fit(X)
    ft = mfe.extract()
    array = ft[1]
    array = np.nan_to_num(array, nan=0)
    result = kdt.query([array], k=1)
    index = result[1][0][0]
    print(index)
    dataset = dataset_sets[index]
    labels = label_sets[index]
    print(f"most similar dataset: n={dataset.shape[0]}, d={dataset.shape[1]}, k={np.unique(labels)}")

    start = time.time()
    for data in dataset_sets:
        y_pred = MeanShift(bandwidth=2).fit_predict(data)
        for metric in MetricCollection.internal_metrics:
            metric.score_metric(data, y_pred)
    print(f"took overall {time.time() - start} seconds")




    #experiment = MetricLearningOfflineExperiment(optimizers=[SMACOptimizer], skip_existing=True)
    #experiment.run_experiments()

    """
    logging.basicConfig(level=logging.DEBUG)

    plt.figure(figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
    plot_num = 1

    optimizer = HyperbandOptimizer
    n_samples = 1000
    n_features = 2
    n_loops = 2
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), [0 for x in range(n_samples)]

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    datasets_per_name = {"circles": noisy_circles, "moons": noisy_moons, "no_struct": no_structure, "blobs": blobs,
                         "aniso": aniso, "varied": varied}
    results = {"dataset": [], "cs": [], "runtime": [], "AMI": [], "ARI": [], "output_dir": [], "Config....": []}

    for i, dataset_name in enumerate(datasets_per_name.keys()):

        X, y = datasets_per_name[dataset_name]
        print(y)
        KMeans().fit_predict(X)

        # define different config spaces for the experiment
        partitional_algos = build_partitional_config_space()
        partitional_algos_dr = build_partitional_dim_reduction_space(n_samples=n_samples, n_features=n_features)
        all_algos = build_all_algos_space()
        all_algos_dr = build_all_algos_dim_reduction_space(n_samples=n_samples, n_features=n_features)

        config_spaces = {
            "part": partitional_algos,
            "part+dr": partitional_algos_dr,
            "all": all_algos,
            "all+dr": all_algos_dr
        }

        for config_space_name in config_spaces.keys():
            results["dataset"].append(dataset_name)
            cs = config_spaces[config_space_name]
            automl_four_clust_instance = optimizer(cs=cs, dataset=X, n_loops=n_loops)
            results["cs"].append(config_space_name)

            # automl_four_clust_instance.get_run_history()

            t0 = time.time()
            result = automl_four_clust_instance.optimize()
            opt_time = time.time() - t0
            results["runtime"].append(opt_time)
            print(automl_four_clust_instance.get_run_history())
            best_configuration = automl_four_clust_instance.get_incumbent()
            y_pred = ClusteringAlgos.execute_algorithm_from_config(X, best_configuration)
            ami_score = MetricCollection.ADJUSTED_MUTUAL.score_metric(data=X, true_labels=y, labels=y_pred)
            ari_score = MetricCollection.ADJUSTED_RAND.score_metric(data=X, true_labels=y, labels=y_pred)
            results["AMI"].append(ami_score)
            results["ARI"].append(ari_score)

            results["output_dir"].append(result.output_dir)

            results["Config...."].append(best_configuration)

            plt.subplot(len(datasets_per_name.keys()), len(config_spaces), plot_num)

            if i == 0:
                plt.title(config_space_name, size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (opt_time)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

    pd.DataFrame(data=results).to_csv("/home/ubuntu/automlclustering/result.csv", sep=";", index=False, decimal=",")
    plt.savefig("/home/ubuntu/automlclustering/results.png")
    """
