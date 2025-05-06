# AutoML4ClustDR
Repository for the prototypical implementation of AutoML4ClustDR, which is built on AutoML4Clust. It adds data reduction functionalities to the original system. Numerous feature selection and sampling-based data reduction methods are available.
Furthermore it can be chosen how the data reduction should be applied. Available application approaches are a naive approach, where the data is reduced before optimization is applied. Next, the Reductionband approach where the dataset is used 
as budget for Hyperband and the subsets for each configurations are generated through the chosen data reduction method. Last, the Substrat approach where the genetic algorithm of the Substrat framework is used to generate a representative 
subset before optimization.


# Running AutoML4ClustDR

Use Python 3.9 to ensure stability. Package requirements for running the code can be found in the `requirements.txt`.

Depending on the size of your dataset you might need to change number of maximum samples allowed per cluster, do so in ClusteringCS.py.

Reductionband might only be runnable for one worker depending on your system

