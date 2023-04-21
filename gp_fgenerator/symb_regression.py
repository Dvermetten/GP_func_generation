
import os
import numpy as np
from .expr2func import expr2func
from .compute_ela import bootstrap_ela, diff_vector
    
#%%
def symb_regr(pset, target_vector, bs_ratio, bs_repeat, list_ela, dict_min, dict_max, dict_weight, dist_metric, verbose, individual, points):
    np.seterr(all='ignore')
    status = 'fail'
    penalty = 1e4
    func, strf = expr2func(individual, pset, points[[0]])
    try:
        list_y = []
        for i in range(len(points)):
            y = func(points[[i]])
            if (y.ndim > 1):
                y = np.mean(y, axis=1)
            list_y.append(y.item())
        y = np.array(list_y)
        y[abs(y) < 1e-12] = 0.0
    except Exception as e:
        if (verbose):
            print(e)
        return status, penalty, strf
    
    if (np.isnan(y).any() or np.isinf(y).any() or np.any(abs(y)>1e20)):
        if (verbose):
            print('invalid')
        return status, penalty, strf
    
    try:
        candidate_vector = bootstrap_ela(points, y, bs_ratio=bs_ratio, bs_repeat=bs_repeat)
        candidate_vector.replace([np.inf, -np.inf], np.nan, inplace=True)
        candidate_vector = candidate_vector.mean(axis=0).to_frame().T
    except Exception as e:
        if (verbose):
            print(e)
        return status, penalty, strf
    
    fitness = diff_vector(candidate_vector, target_vector, list_ela=list_ela, dict_min=dict_min, dict_max=dict_max, dict_weight=dict_weight, dist_metric=dist_metric)
    if (np.isnan(fitness)):
        if (verbose):
            print('dist nan')
        return status, penalty, strf
    status = 'success'
    return status, fitness, strf, candidate_vector
# END DEF