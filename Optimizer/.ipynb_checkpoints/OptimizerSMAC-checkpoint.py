import json
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Dict, Union, Type
import pandas as pd

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from sklearn.cluster import DBSCAN, SpectralClustering, MeanShift
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from smac.facade.hyperband_facade import HB4AC
from smac.facade.smac_ac_facade import SMAC4AC
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_mf_facade import SMAC4MF
from smac.initial_design.sobol_design import SobolDesign
from smac.scenario.scenario import Scenario

from ClusteringCS import ClusteringCS
from CVI import MetricHandler
from Optimizer.smac_function_ import smac_function

from ClusteringCS.ClusteringCS import build_config_space, build_partitional_config_space, \
    build_all_algos_dim_reduction_space, build_partitional_dim_reduction_space, build_all_algos_space, \
    CONFIG_SPACE_MAPPING, build_kmeans_space, execute_algorithm_from_config
from Metrics.MetricHandler import CVICollection
from DataReduction import DataReductionHandler as dr


def preprocess_data(dataset):
    """

    :param dataset:
    :return:
    """
    return StandardScaler().fit_transform(dataset)


def preprocess_3d_data(dataset):
    """

    :param dataset:
    :return:
    """
    return StandardScaler().fit_transform(dataset.reshape(-1, dataset.shape[-1])).reshape(dataset.shape)


class OptimizerHistoryEntry:
    """
    Holds all the information that we want to track for each execution of a configuration during the optimization procedure.
    """

    def __init__(self, runtime, score, labels, config, budget=None, score_name='ARI', additional_metrics=None):
        # Important to note that runtime is the overall wallclock time
        self.runtime = runtime
        self.labels = labels
        self.score = score
        self.configuration = config
        self.score_name = score_name

        if additional_metrics:
            self.additional_metrics = additional_metrics
        else:
            self.additional_metrics = {}

        if budget:
            self.budget = budget
        else:
            self.budget = 0

    def to_dict(self, skip_labels=False):
        if skip_labels:
            return dict({'runtime': self.runtime, self.score_name: self.score, 'budget': self.budget,
                         'config': self.configuration}, **self.additional_metrics)
        else:
            return dict({'runtime': self.runtime, self.score_name: self.score, 'budget': self.budget,
                         'config': self.configuration, 'labels': self.labels}, **self.additional_metrics)

    def __str__(self):
        return f"OptimizerHistoryEntry(runtime={self.runtime}, score={self.score}, labels={len(self.labels)}, " \
               f"budget={self.budget}, config={self.configuration})"

    def __repr__(self):
        return f'OptimizerHistoryEntry[runtime={self.runtime}, score={self.score}, labels={len(self.labels)},' \
               f' budget={self.budget}, config={self.configuration}]'


class AbstractOptimizer(ABC):
    """
       Abstract Wrapper class for all implemented optimization methods.
       Basic purpose is to have convenient way of using the optimizers.
       Therefore, after initializing the optimizer, just by running the
       optimize function the best result found by the optimizer will be obtained.
    """
    n_loops = 50


    def __init__(self, dataset, metric: Union[Type[MetricHandler.Metric], MetricHandler.MLPMetric] = MetricCollection.CALINSKI_HARABASZ,
                 cs: Type[ConfigurationSpace] = None, n_loops=None, budgets=None, output_dir=
                 f"/home/tschecds/automlclustering_old/smac/mario_example/",
                 cut_off_time_minutes=5 * 60, wallclock_limit=30 * 60, true_labels=None, data_reduction: str = None, data_reduction_param: dict = None,      
                 optimization: str = "bayesian", initial_budget = None, max_budget = None ):
        """
        :param dataset: np.array of the dataset (without the labels)
        :param metric: A metric from the MetricCollection. Default is CALINSKI_HARABASZ
        :param cs: ConfigurationSpace object that is used by the optimizer. If not passed, then default is used.
        You can also pass a string, i.e., "partitional" stands for three partitional clustering algorithms (GMM, kMeans, MiniBatchKMeans).
        The value "kmeans" stands for only the kmeans algorithm. The default for the k_range is (2, 200).
        You can also use the file ClusteringCS.py where the strings for the Configspace possibilities are contained.
        :param n_loops: Number of loops that the optimizer performs, i.e., the number of configurations to execute.
        :param output_dir: Output directory where smac stores the runhistory.
        """

        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = f"/home/tschecds/automlclustering_old/smac/mario_example/" #f"/home/ubuntu/automlclustering/smac/{self.get_abbrev()}/"

        if not metric:
            self.metric = MetricCollection.CALINSKI_HARABASZ
        else:
            self.metric = metric
        self.optimization = optimization
        try:
            self.dataset = preprocess_data(dataset)
        except:
            self.dataset = preprocess_3d_data(dataset)
        self.data_reduction = data_reduction
        self.data_reduction_param = data_reduction_param
        if self.optimization == "bayesian":
            
            if self.data_reduction is not None:
            
                self.dataset = dr.DataReductionHandler(self.dataset, self.data_reduction, self.data_reduction_param).data_reduction_main_method()
                #new_dataset = dr.data_reduction_main_method(dataset, data_reduction, data_reduction_param)

        if not n_loops:
            # set default budget
            self.n_loops = 60
        else:
            self.n_loops = n_loops

        if not cs:
            # n_samples = self.dataset.shape[0]
            # n_features = self.dataset.shape[1]
            # build default config space
            self.cs = build_kmeans_space()
        else:
            if isinstance(cs, str) and cs in CONFIG_SPACE_MAPPING:
                self.cs = CONFIG_SPACE_MAPPING[cs]
            elif isinstance(cs, ConfigurationSpace):
                self.cs = cs

        if not initial_budget:
            # hard code budgets and eta for Hyperband, don't know a better solution at the moment.
            self.budgets = [1, 3, 10]
            self.eta = 10
            # self.budgets = [2, 10]
        else:
            self.initial_budget = initial_budget
            self.max_budget = max_budget

        self.true_labels = true_labels
        self.optimizer_result = None
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.trajectory_dict_list = []

        # default cutoff time is 10 minutes. The cutoff time has to be given in seconds!
        self.cutoff_time = cut_off_time_minutes
        # maximum runtime for the whole optimization procedure in seconds. Default is 60 minutes (60 * 60 seconds)
        self.wallclock_limit = wallclock_limit

    def optimize(self):
        return self.optimizer_result

    @staticmethod
    @abstractmethod
    def get_name():
        pass

    @classmethod
    def get_abbrev(cls):
        return OPT_ABBREVS[cls.get_name()]

    def get_config_space(self):
        return self.cs


class SMACOptimizer(AbstractOptimizer):
    """
        State-of-the-Art Bayesian Optimizer.
         Can be configured to use Random Forests (TPE) or Gaussian Processes.
        However, Gaussian Processes are much slower and only work for low-dimensional parameter spaces.
        Due to this, the default implementation uses Random Forests.
    """
    #new_dataset = None
    
    def __init__(self, dataset, wallclock_limit=30*60, metric=None, n_loops=None,  smac=SMAC4HPO, cs=None,
                 ouput_dir=None, true_labels=None, data_reduction: str = None, data_reduction_param: dict = None, optimization: str = "bayesian",  
                 initial_budget = None, max_budget = None ):
        super().__init__(dataset=dataset, metric=metric, cs=cs, n_loops=n_loops, output_dir=ouput_dir,
                         true_labels=true_labels, wallclock_limit=wallclock_limit, data_reduction=data_reduction, data_reduction_param=data_reduction_param,
                        optimization=optimization, initial_budget = initial_budget, max_budget = max_budget)


        self.data_reduction = data_reduction
        self.data_reduction_param = data_reduction_param
        #if optimization == "bayesian":
        self.smac_algo = smac
        #else:
            #self.smac_algo = HB4AC
        #self.smac = None

    def optimize(self, initial_configs=None) -> Union[SMAC4HPO, SMAC4AC, SMAC4BB]:
        
        """
        Runs the optimization procedure. Requires that the optimizer is instantiated with a dataset and
        a suitable config space. The procedure returns the smac optimizer. However, you probably want to use the other
        methods the optimizer offers to get either the best configuration (get_incumbent)
        or to get the history (get_run_history).

        :param initial_configs: Initial configurations that the optimizer first selects in the first
        n_initial_configs loops. This can be useful for using meta-learning.
        :return:
        """
        

        # subsamples of datasets, key ist the budget and value are the indices of the samples
        # not used for SMACOptimizer but for Hyperband and BOHB
        #budget_indices = {b: np.random.choice(len(self.dataset), int(len(self.dataset) * int(b) / 10)) for b in
                          #self.budgets}
        
        if self.data_reduction and self.optimization == "hyperband":

            kwargs = {"data_reduction": self.data_reduction, "data_reduction_param": self.data_reduction_param, 'max_budget': self.max_budget}
            self.cut_off_time_minutes=3 * 60
            self.wallclock_limit=5 * 60
            tae_algorithm = partial(smac_function, optimizer_instance=self, **kwargs)
        else: 

            kwargs = {"doesn't_matter": "at_all"}
        tae_algorithm = partial(smac_function, optimizer_instance=self, **kwargs) #budget_indices=budget_indices)

        # Scenario object
        
            
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": self.n_loops,  # max. number of function evaluations;
                             "cs": self.cs,  # configuration space
                             "deterministic": "true",
                             "output_dir": self.output_dir,
                             "cutoff": self.cutoff_time,
                             # max duration to run the optimization (in seconds)
                             "wallclock-limit": self.wallclock_limit,
                             "run_id": str(uuid.uuid4)
                             })

        n_workers = 1 if self.get_name() == SMACOptimizer.get_name() else 1
        self.smac = self.smac_algo(scenario=scenario,
                                   tae_runner=tae_algorithm,
                                   initial_configurations=None if not initial_configs else initial_configs,
                                   initial_design=None if initial_configs else SobolDesign,
                                   # intensifier=SuccessiveHalving,
                                   intensifier_kwargs=self.get_intensifier_kwargs(),
                                   n_jobs=n_workers
                                   )
        print(self.smac)
        self.smac.optimize()

        return self.smac

    def get_incumbent(self) -> Configuration:
        return self.smac.solver.incumbent

    def set_trajectory(self, trajectory_dict_list):
        self.trajectory_dict_list = trajectory_dict_list

    def get_trajectory(self) -> List[Dict]:
        """
        Returns a list of dictionary which kept track of the "trajectory", i.e., of the incumbents.
        So how long it took to get the incumbent etc. Yet this only tracks the history for the best configuration
        at each loop, not the whole history.
        :return: List of dictionaries
        """
        if not self.trajectory_dict_list:
            self.trajectory_dict_list = [x._asdict() for x in self.smac.get_trajectory()]
        return self.trajectory_dict_list

    def get_run_history(self) -> List[OptimizerHistoryEntry]:
        """
        Returns a list of OptimizerHistoryEntry object, which contain the results of each configuration that the
         optimizer executed.
        :return:
        """
        if not self.optimizer_result:
            self.parse_result()

        return self.optimizer_result

    def get_run_hitory_list_dicts(self, skip_labels=False) -> List[Dict]:
        """
        Uses get_run_history to return a list of dictionaries where each dictionary in the list has the same keys.
        The reason for this is to easy export this to a csv file with a dataframe for example.
        :return:
        """
        if not self.optimizer_result:
            self.parse_result()
        history = self.get_run_history()
        return [opt_history_entry.to_dict(skip_labels=skip_labels) for opt_history_entry in history]

    def get_runhistory_df(self):
        return pd.DataFrame(data=self.get_run_hitory_list_dicts())

    def parse_result(self):
        """
        Should only be called after the optimize method was called. This method parses the "runhistory.json" and stores
        the information in the optimizer_result. The result can be get by calling the get_runhistory() method.
        """
        out_dir = self.smac.output_dir
        history = []

        with open('{}/runhistory.json'.format(out_dir)) as json_file:
            json_data = json.load(json_file)
            data = json_data['data']
            configs = json_data['configs']

            for runs in data:
                # parse budget
                budget = runs[0][3]

                conf_id = runs[0][0]
                config = configs[str(conf_id)]

                # parse metric score and runtime of evaluating the algorithm+metric
                run = runs[1]
                score = run[0]
                optimizer_time = run[1]

                # parse addititonal info like labels
                add_info = run[5]
                if "labels" in add_info:
                    y_pred = add_info["labels"]
                else:
                    # can happen that labels are not there, because the execution was cutoff
                    y_pred = [-2 for _ in self.dataset]

                additional_metrics = {}
                for internal_metric in MetricCollection.internal_metrics:
                    if internal_metric.get_abbrev() in add_info:
                        metric_score = add_info[internal_metric.get_abbrev()]
                        additional_metrics[internal_metric.get_abbrev()] = metric_score

                # create entry object and save to list
                entry = OptimizerHistoryEntry(labels=y_pred, budget=budget, runtime=optimizer_time, score=score,
                                              config=config, additional_metrics=additional_metrics,
                                              score_name=self.metric.get_abbrev())
                history.append(entry)

        self.optimizer_result = history

    def get_intensifier_kwargs(self):
        return {}

    @staticmethod
    def get_name():
        return "SMAC"


class HyperbandOptimizer(SMACOptimizer):
    def __init__(self, dataset, metric=None, n_loops=None,
                 cs=None, smac=HB4AC, output_dir=None, true_labels=None, data_reduction: str = None, data_reduction_param: dict = None, initial_budget = None,                    max_budget = None, optimization: str = "hyperband"):
        super().__init__(dataset=dataset, metric=metric, cs=cs,
                         n_loops=n_loops, smac=smac, ouput_dir=output_dir, true_labels=true_labels, data_reduction=data_reduction, 
                         data_reduction_param=data_reduction_param, initial_budget = initial_budget, max_budget = max_budget, optimization=optimization)
        
        
    def get_intensifier_kwargs(self):
        return {'initial_budget': self.initial_budget, 'max_budget': self.max_budget, 'eta': 3}

    @staticmethod
    def get_name():
        return "Hyperband"


class BOHBOptimizer(HyperbandOptimizer):
    def __init__(self, dataset, metric=None, n_loops=None,
                 cs=None, smac=SMAC4MF, output_dir=None, true_labels=None):
        super().__init__(dataset=dataset, metric=metric, cs=cs, n_loops=n_loops, smac=smac, output_dir=output_dir,
                         true_labels=true_labels)

    @staticmethod
    def get_name():
        return "BOHB"


class RandomOptimizer(AbstractOptimizer):

    def __init__(self, dataset, metric=None, n_loops=None,
                 cs=None, true_labels=None):
        super().__init__(dataset=dataset, metric=metric, cs=cs, n_loops=n_loops, true_labels=true_labels)
        self.incumbent = None

    def optimize(self, initial_configs=None):
        if initial_configs is None:
            initial_configs = []
        best_score = np.inf

        # sample n_loops - |init_configs| from cs randomly
        configs = self.cs.sample_configuration(size=self.n_loops - len(initial_configs))

        # extend configs list with init_configs
        configs.extend(initial_configs)
        self.incumbent = None

        for config in configs:
            score, _ = smac_function(config, self.dataset, metric=self.metric)
            if score < best_score:
                best_score = score
                self.incumbent = config

        return best_score

    def get_incumbent(self):
        return self.incumbent

    @staticmethod
    def get_name():
        return "Random"


OPT_ABBREVS = {SMACOptimizer.get_name(): "BO",
               HyperbandOptimizer.get_name(): "HB",
               BOHBOptimizer.get_name(): "BOHB",
               RandomOptimizer.get_name(): "RS"}

if __name__ =='__main__':
    np.random.seed(0)
    cs = ClusteringCS.build_all_algos_space()

    t0 = time.time()
    dbscan = DBSCAN(eps=0.7, min_samples=20)
    n_samples = 10000
    data = make_blobs(n_samples=n_samples, n_features=30,random_state=1234)
    X = data[0]
    y = data[1]
    X = StandardScaler().fit_transform(X)
    #y_pred = dbscan.fit_predict(X)
    #optimizer = SMACOptimizer(true_labels=y, n_loops=100, metric=metric, cs=cs,
    #                          dataset=X)
    #optimizer.optimize()

    #print(optimizer.get_incumbent())

    #inc = optimizer.get_incumbent()
    #y_pred = ClusteringCS.execute_algorithm_from_config(X, inc)
    start = time.time()
    y_pred = SpectralClustering(n_clusters=200).fit_predict(X)
    print(f"Spectral Clustering took: {time.time() - start}")
    # optimizer.get_trajectory()

    print(np.unique(y_pred))
    # print(y_pred)
    for metric in MetricCollection.internal_metrics:
        start = time.time()
        metric.score_metric(X, y_pred)
        print(f"Execution of metric {metric.get_abbrev()} took {time.time() - start}s")
    """
    import matplotlib.pyplot as plt
    from itertools import cycle, islice

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y_pred) + 1))))

    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
    plt.show()
    t1 = time.time() - t0
    """
    ari = MetricCollection.ADJUSTED_RAND.score_metric(data=X, labels=y_pred, true_labels=y)
    print(f"took overall {time.time() - t0}")
    ami = MetricCollection.ADJUSTED_MUTUAL.score_metric(data=X, labels=y_pred, true_labels=y)
    dbcv = MetricCollection.DENSITY_BASED_VALIDATION.score_metric(data=X, labels=y_pred)
    print(f"DBCV score is: {dbcv}")
    print(f"ari is: {ari}")
    print(f"ami is: {ami}")


