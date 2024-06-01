from skfeature.function.similarity_based import SPEC as Spec
from coreset import coreset
from sklearn.feature_selection import VarianceThreshold as vt
import numpy as np
import pandas as pd
import math
import random
from SubStrat.summary_algorithm.genetic_sub_algorithm import GeneticSubAlgorithmn
import sys
sys.path.insert(1,"/home/tschecds/automlclustering_old/src/DataReduction/Laplacian_Score_Feature_Selection")

import lp_score


"""
Responsible for everything related to optional data reduction.
It contains all names of data reduction methods, what attributes they need, a class that can yield any scores and rankings claculated and functions that execute the metrics and return the resulting subset
"""

def data_reduction_names():
        """
        Returns a list of the names of all available data_reduction_methods.
        :return: List of names
        """
             
        names = ["UniformRandomSampling", "LightWeightCoreset", "LaplacianScore", "SPEC", "Substrat", "Variance", "UniformRandomFeatureSelection",  "ProTras"]
    
        return names


def execute_data_reduction(dataset, data_reduction, data_reduction_param):

    
    if data_reduction == "UniformRandomSampling":
        
        sample_size = data_reduction_param.get("sample_size")
        reduction_instance = UniformRandomSampling(dataset, sample_size)
        subset = reduction_instance.reduce()

    if data_reduction == "LightWeightCoreset":
        
        sample_size = data_reduction_param.get("sample_size")
        reduction_instance = LightWeightCoreset(dataset, sample_size)
        subset = reduction_instance.reduce()
        
    if data_reduction == "ProTras":
        
        sample_size = data_reduction_param.get("sample_size")
        reduction_instance = ProTras(dataset, sample_size)
        subset = reduction_instance.reduce()

    if data_reduction == "UniformRandomFeatureSelection":
        
        feature_number = data_reduction_param.get("feature_number")
        reduction_instance = UniformRandomFeatureSelection(dataset, feature_number)
        subset = reduction_instance.reduce()
        
    if data_reduction == "Variance":
        
        feature_number = data_reduction_param.get("feature_number")
        reduction_instance = Variance(dataset, feature_number)
        subset = reduction_instance.reduce()

    if data_reduction == "LaplacianScore":
        
        feature_number = data_reduction_param.get("feature_number")
        t_param = data_reduction_param.get("t_param")
        neighbour_size = data_reduction_param.get("neighbour_size")
        reduction_instance = LaplacianScore(dataset, feature_number, t_param, neighbour_size)
        subset = reduction_instance.reduce()

    if data_reduction == "SPEC":
        
        feature_number = data_reduction_param.get("feature_number")
        spec_function = data_reduction_param.get("spec_function")
        reduction_instance = SPEC(dataset, feature_number, spec_function)
        subset = reduction_instance.reduce()

    if data_reduction == "Substrat":
        
        feature_number = data_reduction_param.get("feature_number")
        sample_size = data_reduction_param.get("sample_size")
        fitness_function = data_reduction_param.get("fitness_function")
        #kwargs = data_reduction_param.get("kwargs")
        reduction_instance = Substrat(dataset, feature_number, sample_size, fitness_function)
        subset = reduction_instance.reduce()

    return subset



def uniform_random_sampling(X, n):
    
    sample = []
    sampling_indexes = np.random.randint(low=0, high=len(X), size=n)
    for i in sampling_indexes:
        sample.append(X[i])
    sample =  np.asarray(sample)
    print(sample)
    return sample

def uniform_random_feature_selection(X, n):
    
    columns = X.shape[1]
    sampling_indexes = np.random.randint(low=0, high=columns, size=n)
    for i in range(0,columns):
        if i not in sampling_indexes:
            sample = np.delete(X, i, 1)
    
    return sample

def l1_dist(value1: float,
            value2: float):
        return math.fabs(value1 - value2)

def rank(scores):

    ranking = scores[scores[:, 0].argsort()]
    
    return ranking


def add_dimension_indices(scores):
    N= len(scores)
    scores = scores[:, np.newaxis]
    b = np.zeros((N,2))
    b[:,:-1] = scores
    scores= b
    for i in range(N):
        scores[i][1] = i
    return scores


#get coreset of size n using modified protras algorithm
def protras_size_parameter(X, n):
    
    """
    Algorithm based on:
    L. H. Trang, N. Le Hoang and T. K. Dang, "A Farthest First Traversal based Sampling Algorithm for k-clustering," 2020 14th International Conference on 
    Ubiquitous Information Management and Communication (IMCOM), Taichung, Taiwan, 2020, pp. 1-6, doi: 10.1109/IMCOM48794.2020.9001738.

    """
    iteration=0
    distances=[]
    group_centers=[]
    #choose random initial instance
    group_centers.append(random.randint(0,len(X)))
    while(len(group_centers)<n):
        if iteration==0:
            distances=np.linalg.norm(X-X.iloc[group_centers[iteration]],axis=1,keepdims=True)
        else:
            new_distances=np.linalg.norm(X-X.iloc[group_centers[iteration]],axis=1,keepdims=True)
            distances=np.concatenate((distances,new_distances),axis=1)
        min_distances=np.argmin(distances,axis=1)
        max_wd = 0
        #assign None to instance if no suitable instance is found  
        instance = None
        for i in range(iteration+1):
            indices=np.where(min_distances==i)[0]
            p=np.max(distances[indices,i])
            max_val=p*len(indices)
            if max_wd<max_val:
                max_wd=p
                instance=indices[np.argmax(distances[indices,i])]
        group_centers.append(instance)    
        iteration = iteration+1
    return np.array(X.iloc[group_centers])
        
        


class UniformRandomSampling:

    
    def __init__(self, dataset, sample_size: int):
        self.dataset= dataset
        self.sample_size= sample_size

    def get_name(self):

        return self__name__

    def reduce(self):

        subset = uniform_random_sampling(self.dataset, self.sample_size)

        return subset

class LightWeightCoreset:

    
    def __init__(self, dataset, sample_size: int):
        self.dataset= dataset
        self.sample_size= sample_size

    
    def get_name(self):

        return self__name__

    def reduce(self):

        subset = coreset.generate(self.dataset, self.sample_size)

        return subset[0]

class ProTras:

    
    def __init__(self, dataset, sample_size: int):
        self.dataset= dataset
        self.sample_size= sample_size

    
    def get_name(self):

        return self__name__

    def reduce(self):
        col_list = []
        for i in range(0, self.dataset.shape[1]):
            col = str(i)
            col_list.append(col)
        pd_dataset =  pd.DataFrame(self.dataset, columns=col_list)
        subset = protras_size_parameter(pd_dataset, self.sample_size)

        return subset

class Variance:

    
    def __init__(self, dataset, feature_number: int = 7):
        self.dataset= dataset
        self.feature_number= feature_number


    def get_name(self):

        return self__name__

    def score(self):

        data = self.dataset.to_numpy()
        scores = vt.selector.fit(data)
        scores = vt.scores.variances_

        return sum(scores)
        

    def reduce(self):
        indices= []
        selector = vt() #using sklearn Variancethreshold to get feature scores
        scores = selector.fit(self.dataset)
        scores = scores.variances_
        scores = add_dimension_indices(scores)
        ranking = rank(scores)
        reduced_ranking = ranking[-(len(ranking)-self.feature_number):] #only leave features that are to be deleted
        for i in range(len(reduced_ranking)):
            indices.append(reduced_ranking[i][1])
            indices[i] = int(indices[i])
        subset = np.delete(self.dataset, indices, 1)

        return subset


class UniformRandomFeatureSelection:

    """
    Class for uniform random feature selection
        
    """

    
    def __init__(self, dataset, feature_number: int):
        self.dataset= dataset
        self.feature_number= feature_number

        """
        :param dataset: np.array of the dataset (without the labels)
        :param feature_number: number of features that are to remain
        
        """

    def get_name(self):

        return self__name__

    def reduce(self):

        subset = uniform_random_feature_selection(self.dataset, self.feature_number)

        return subset


class LaplacianScore:

    """
    Class for calculation of Laplacian Score of all features and selection of features based on the results
        
    """
    
    def __init__(self, dataset, feature_number: int = 7, t_param = 625, neighbour_size: int = 5):
        self.dataset=dataset
        self.feature_number=feature_number
        self.t_param=t_param
        self.neighbour_size=neighbour_size

        """
        :param dataset: np.array of the dataset (without the labels)
        :param feature_number: number of features that are to remain
        :param t_param: Value used in the weighting function, default is the RBF Kernel with t=25^2
        :param neighbour_size: k used for the knn-algorithm used to build the similarity graph
        
        """

    def get_name(self):

        return self__name__

    def score(self):
        
        scores = lp_score.LaplacianScore(data, t_param=self.t_param, neighbour_size=self.neighbour_size)

        return scores

    def reduce(self):
        indices= []
        scores = lp_score.LaplacianScore(self.dataset, t_param=self.t_param, neighbour_size=self.neighbour_size)
        #add dimension indices before reordering array
        scores = add_dimension_indices(scores)
        ranking = rank(scores)
        #the smaller the laplacian score is, the more important the feature is
        reduced_ranking = ranking[-(len(ranking)-self.feature_number):] #only leave features that are to be deleted
        for i in range(len(reduced_ranking)):
            indices.append(reduced_ranking[i][1])
            indices[i] = int(indices[i])
        subset = np.delete(self.dataset, indices, 1)
        
        return subset

#genetic algorithm needs pd dataframe, fs-methods need np.array, changing between them might make substrat useless runtime-wise
class SPEC:

    """
    Class for calculation of SPEC functions' scores of all features and selection of features based on the results
        
    """
    
    def __init__(self, dataset, feature_number: int=7, spec_function=-1):
        self.dataset= dataset
        self.feature_number= feature_number
        self.spec_function= spec_function

        """
        :param dataset: np.array of the dataset (without the labels)
        :param feature_number: number of features that are to remain
        :param spec_function: determines which of the two spec_functions is used, possible values are -1 and 0
        
        """
        
    def get_name(self):

        return self__name__

    def score(self):
        
        scores = Spec.spec(data)

        return scores
        
    #add the different ranking functions to choose
    def reduce(self):
        indices= []
        scores = Spec.spec(self.dataset, kwargs = {"style": self.spec_function})
        scores = add_dimension_indices(scores)
        ranking = rank(scores)
        reduced_ranking = ranking[-(len(ranking)-self.feature_number):] #only leave features that are to be deleted
        for i in range(len(reduced_ranking)):
            indices.append(reduced_ranking[i][1])
            indices[i] = int(indices[i])
        subset = np.delete(self.dataset, indices, 1)

        return subset

#extra class for substrat needed since dataset is only passed to the score function of a fitness class in GeneticAlgorithm of Substrat

class Spec_Fitness:

    
    def __init__(self, *args, **kwargs):
        
        if kwargs:
            self.kwargs = kwargs['kwargs']
            print(self.kwargs)
        """
        :param kwargs: set which SPEC function is to be used {"style": -1} 
        or {"style": 0} if no kwargs are provided the SPEC framework will default
        to using {"style": 0} 

        call like this when running Substrat "fitness_function": Spec_Fitness(kwargs= {"style": -1}
        """
    
    def score(self, full_ds: pd.DataFrame,
                     sub_ds: pd.DataFrame):
        
        data = full_ds.to_numpy()
        print(self.kwargs)
        full_ds_score = sum(Spec.spec(data, kwargs = self.kwargs))
        print(full_ds_score)
        data = sub_ds.to_numpy()
        sub_ds_score = sum(Spec.spec(data, kwargs = self.kwargs))
        
        dist = l1_dist(full_ds_score, sub_ds_score)

        print(dist)
        return 1.0 / dist if dist != 0 else float('inf')


class Spec_Fitness_2:

    
    def __init__(self, *args, **kwargs):
        pass

    
    def score(self, full_ds: pd.DataFrame,
                     sub_ds: pd.DataFrame):
    
        data = full_ds.to_numpy()
        full_ds_score = sum(Spec.spec(data), kwargs = {"style": 0})
        data = sub_ds.to_numpy()
        sub_ds_score = sum(Spec.spec(data), kwargs = {"style": 0})
        
        dist = l1_dist(full_ds_score, sub_ds_score)

        print(dist)
        return 1.0 / dist if dist != 0 else float('inf')

class Spec_Fitness_1:

    
    def __init__(self, *args, **kwargs):
        pass

    
    def score(self, full_ds: pd.DataFrame,
                     sub_ds: pd.DataFrame):
    
        data = full_ds.to_numpy()
        full_ds_score = sum(Spec.spec(data), kwargs = {"style": -1})
        data = sub_ds.to_numpy()
        sub_ds_score = sum(Spec.spec(data), kwargs = {"style": -1})
        
        dist = l1_dist(full_ds_score, sub_ds_score)

        print(dist)
        return 1.0 / dist if dist != 0 else float('inf')


class Lp_Fitness:

    
    def __init__(self, *args, **kwargs):
        pass

    
    def score(self, full_ds: pd.DataFrame,
                     sub_ds: pd.DataFrame):

        data = full_ds.to_numpy()
        full_ds_score = sum(lp_score.LaplacianScore(data, t_param = 625, neighbour_size = 5))
        data = sub_ds.to_numpy()
        sub_ds_score = sum(lp_score.LaplacianScore(data, t_param = 625, neighbour_size = 5))

        
        dist = l1_dist(full_ds_score, sub_ds_score)

        
        return 1.0 / dist if dist != 0 else float('inf')

class Variance_Fitness:

    
    def __init__(self, *args, **kwargs):
        pass

    
    def score(self, full_ds: pd.DataFrame,
                     sub_ds: pd.DataFrame):
        selector = vt()
        data = full_ds.to_numpy()
        full_ds_score = selector.fit(data)
        full_ds_score = sum(full_ds_score.variances_)
        data = sub_ds.to_numpy()
        sub_ds_score = selector.fit(data)
        sub_ds_score = sum(sub_ds_score.variances_)

        
        dist = l1_dist(full_ds_score, sub_ds_score)
        
        return 1.0 / dist if dist != 0 else float('inf')

    

class Substrat:


    def __init__(self, dataset, feature_number: int, sample_size: int, fitness_function, **kwargs):
        self.dataset= dataset
        self.feature_number= feature_number
        self.sample_size= sample_size
        self.fitness_function= fitness_function
        


    def get_name(self):

        return self__name__

    def reduce(self):
        col_list = []
        for i in range(0, self.dataset.shape[1]):
            col = str(i)
            col_list.append(col)
        pd_dataset =  pd.DataFrame(self.dataset, columns=col_list)
        reduction_instance = GeneticSubAlgorithmn(dataset = pd_dataset, target_column_name = "0", sub_col_size = self.feature_number, 
                 sub_row_size = self.sample_size, population_size = 10, fitness = self.fitness_function, mutation_rate = 0.04, 
                 num_generation = 1, stagnation_limit = 20, time_limit = float(5*60))
        subset = reduction_instance.run()
        subset = subset.to_numpy()
        
        return subset


class DataReductionHandler:

    def __init__(self, dataset, data_reduction: str, data_reduction_param):
        self.dataset= dataset
        self.data_reduction= data_reduction
        self.data_reduction_param= data_reduction_param
    
    
    #TO-DO: can be implemented into execution method, as try block around all if's, automatically throws exception when it's not in list
    def data_reduction_main_method(self):
        
        """
        Checks input validity and either calls execution method that returns subset or 
        an error message if the input was invalid.
        :return:
        """
        
        names = data_reduction_names()
    
        if self.data_reduction in names:
            
            subset = execute_data_reduction(self.dataset, self.data_reduction, self.data_reduction_param)
            
            return subset
    
        else:
            return print("given data reduction method not in name list")