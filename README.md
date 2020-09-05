# StabilityRankings_AD
A Ranking Stability Measure for Quantifying the Robustness of Anomaly Detection Methods

`StabilityRankings_AD` (Ranking Stability Measure) is a GitHub repository containing the algorithm for estimating the **ranking stability measure** [1].
It refers to the paper titled *A Ranking Stability Measure for Quantifying the Robustness of Anomaly Detection Methods*. 

## Abstract
Anomaly detection attempts to learn models from data that can detect anomalous examples in the data. However, naturally occurring variations in the data impact the model that is learned and thus which examples it will predict to be anomalies. Ideally, an anomaly detection method should be robust to such small changes in the data. Hence, this paper introduces a *ranking stability measure* that quantifies the robustness of any anomaly detector’s predictions by looking at how consistently it ranks examples in terms of their anomalousness. Our experiments investigate the performance of this stability measure under different data perturbation schemes. In addition, they show how the stability measure can complement traditional anomaly detection performance measures, such as *area under the ROC curve* or *average precision*, to quantify the behaviour of different anomaly detection methods.

## Contents and usage

The repository contains:
- stability.py, the function that allows to get stability measure;
- Notebook.ipynb, a notebook showing how to use the function on an artificial dataset;

To use this algorithm, import the github repository or simply download the files. You can also find the benchmark datasets inside the folder Benchmark_Datasets or at this [[link](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/)].

## A Stability Measure to Quantify the Robustness of Any Anomaly Detection Method

Given a training set **Xtr**, a test set **Xte**, an anomaly detector **h** and a contamination factor <img src="https://render.githubusercontent.com/render/math?math=\gamma">, the algorithm estimates the model stability in three steps. First, it *randomly draws subsets* from the training set **Xtr** to simulate slight changes in the set of available training examples. Each time, the model **h** is retrained and a ranking over **Xte** gets constructed. Second, it *assigns a stability score* to each test set example by taking into account both the variance and the range of its normalized rank positions. Finally, the algorithm *aggregates the stability scores* of all test set examples to obtain the final ranking stability measure, quantifying the robustness of the model **h** on the considered dataset. The method works with **any anomaly detector**, since it deals with the ranking positions depending on the anomaly scores, where the top positions belong to the most normal examples (with lowest scores).

The method has *two choices* for giving the shape to the Beta distribution. By setting *beta_flavor* equal to 1, the assumption is that the area under the Beta distribution is the same in both the intervals <img src="https://render.githubusercontent.com/render/math?math=[0, \psi]"> and <img src="https://render.githubusercontent.com/render/math?math=[\psi,1]">, where <img src="https://render.githubusercontent.com/render/math?math=\psi \in (0,1)">. On the other hand, setting *beta_flavor* equal to 2 leads to set the width of the Beta distributon such that <img src="https://render.githubusercontent.com/render/math?math=\psi">% of the mass of the distribution falls in the region <img src="https://render.githubusercontent.com/render/math?math=[1 - 2 \gamma, 1]">.

The algorithm can be applied as follows:

```python
from pyod.models.knn import KNN
from stability import *

# Estimate the contamination factor (which has to be given), for instance with
gamma = sum(y)/len(y)      # where y is the label vector (1 for anomaly, 0 for normal);

# Build an anomaly detector h (for instance, here we use kNNO)
knno = KNN(n_neighbors = 5, contamination = gamma)

# Compute the ranking stability measure under UNIFORM sampling:
knno_stab_unif, knno_inst_unif = stability_measure(Xtr, Xte, knno, gamma, unif = True)

# Compute the ranking stability measure under BIAS sampling:
knno_stab_bias, knno_inst_bias = stability_measure(Xtr, Xte, knno, gamma, unif = False)
```

## Dependencies

The `stability` function requires the following python packages to be used:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Pandas](https://pandas.pydata.org/)

## Contact

Contact the author of the paper: [lorenzo.perini@kuleuven.be](mailto:lorenzo.perini@kuleuven.be).


## References

[1] Perini, L., Galvin, C., Vercruyssen, V.: *A Ranking Stability Measure for Quantifyingthe Robustness of Anomaly Detection Methods.* In: 2nd Workshop on Evaluation and Experimental Design in Data Mining and Machine Learning @ ECML/PKDD (2020).
