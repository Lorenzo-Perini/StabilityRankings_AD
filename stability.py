import random, sys
import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
from scipy.optimize import Bounds, LinearConstraint, minimize
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
import pandas as pd

def stability_measure(Xtr, Xte, model, gamma,
        unif = True,            # pick True or False
        iterations=500,
        psi = 0.8,
        beta_flavor= 2,          # pick from: 1, 2
        subset_low=0.25,
        subset_high=0.75,
        intermediate_scores=False):
        
    """ 
    Parameters
    ----------
    Xtr                 : np.array of shape (n_samples,n_features) containing the training set;
    Xte                 : np.array of shape (m_samples,n_features) containing the test set;
    model               : object containing the anomaly detector;
    gamma               : float containing the contamination factor, i.e. the expected proportion of anomalies;
    unif                : bool selecting the case to exploit: If True, uniform sampling is selected, if False, biased sampling is used;
    iterations          : int regarding the number of iterations;
    psi                 : float in [0, 1], the hyperparameter controlling the shape of the beta distribution;
    beta_flavor         : int equal to either 1 or 2 selecting the way the beta distribution parameters are chosen;
    subset_low          : float containing the lower bound of subsample size as percent of training length;
    subset_high         : float containing the upper bound of subsample size as percent of training length;
    intermediate_scores : bool which selects whether also to compute all intermediate stability scores. Default = False.

    Returns
    -------
    S                   : float representing the stability measure;
    IS                  : float representing the instability measure.
    """
    
    np.random.seed(331)
    ntr, _ = Xtr.shape
    nte, _ = Xte.shape
    
    n_clust = 10
    # sample weights
    if not unif:
        cluster_labels = KMeans(n_clusters = n_clust, random_state=331).fit(Xtr).labels_ +1
    else:
        weights = np.ones(ntr) * (1 / ntr)

    if psi == 0:
        psi = max(0.51, 1 - (gamma + 0.05))
    
    # sample weights
    norm = np.ones(ntr) * (1 / ntr)

    # compute the rankings over the test set
    point_rankings = np.zeros((nte, iterations), dtype=np.float)
    for i in range(iterations):
        if not unif:
            biased_weights = {}
            for w in range(1,n_clust+1):
                biased_weights[w] = np.random.randint(1,100)
            weights = np.asarray([biased_weights[w] for w in cluster_labels])
            weights = weights/sum(weights)
        # draw subsample
        subsample_size = np.random.randint(int(ntr *subset_low), int(ntr * subset_high))
        sample_indices = np.random.choice(np.arange(ntr), size=subsample_size, p=weights, replace=False)
        Xs = Xtr[sample_indices, :]

        # fit and predict model
        model.fit(Xs)
        probs = model.predict_proba(Xte, method='unify')
        anom_probs = np.nan_to_num(probs)[:, 1]

        # construct test set rankings
        sorted_ixs = np.argsort(anom_probs)  # first index = lowest score
        for ii, si in enumerate(sorted_ixs):
            point_rankings[si, i] = ii + 1

    # normalize rankings
    point_rankings = point_rankings / nte  # lower rank = more normal

    if beta_flavor == 1:
        # The area of the Beta distribution is the same in the intervals [0, psi] and [psi, 1]
        beta_param = float((1/(psi + gamma -1))*(2*gamma -1 - gamma/3 + psi*((3 - 4*gamma)/3)))
        alpha_param = float(beta_param*((1-gamma)/gamma) + (2*gamma -1)/gamma)
        
    elif beta_flavor == 2:
        # the width of the beta distribution is set such that psi percent of the mass
        # of the distribution falls in the region [1 - 2 * gamma , 1]
        
        # optimization function
        def f(p):
            return ((1.0 - psi) - beta.cdf(1.0 - 2 * gamma, p[0], p[1])) ** 2
        # bounds
        bounds = Bounds([1.0, 1.0], [np.inf, np.inf])
        # linear constraint
        linear_constraint = LinearConstraint([[gamma, gamma - 1.0]], [2*gamma - 1.0], [2*gamma - 1.0])
        # optimize
        p0 = np.array([1.0, 1.0])
        res = minimize(f, p0, method='trust-constr', constraints=[linear_constraint], options={'verbose': 0}, bounds=bounds)
        alpha_param = res.x[0]
        beta_param = res.x[1]
    else:
        print('Wrong choice! Pick a better one!')
        sys.exit()

    # compute the stability score for multiple iterations
    random_stdev = np.sqrt((nte+1)*(nte-1)/(12*nte**2))
    stability_scores = []
    lower = 2 if intermediate_scores else iterations
    for i in range(lower, iterations + 1):
        # point stabilities
        point_stabilities = np.zeros(nte, dtype=float)
        for ii in range(nte):
            p_min = np.min(point_rankings[ii, :i])
            p_max = np.max(point_rankings[ii, :i])
            p_std = np.std(point_rankings[ii, :i])
            p_area = beta.cdf(p_max, alpha_param, beta_param) - beta.cdf(p_min, alpha_param, beta_param)
            point_stabilities[ii] = p_area * p_std

        # aggregated stability
        stability_scores.append(np.mean(np.minimum(1, point_stabilities/random_stdev)))

    stability_scores = 1.0 - np.array(stability_scores)
    instability_scores = 1.0 - stability_scores
    
    if intermediate_scores:
        return stability_scores , instability_scores
    else:
        return stability_scores[0], instability_scores[0]