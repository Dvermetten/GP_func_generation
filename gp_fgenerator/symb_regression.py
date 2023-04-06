

import numpy as np
from .expr2func import expr2func
from .compute_ela import bootstrap_ela, diff_vector



    
#%%
def symb_regr(pset, target_vector, dist_metric, bs_repeat, individual, points):
    np.seterr(all='ignore')
    penalty = 1e4
    func = expr2func(individual, pset, points)
    try:
        y = func(points)
        if (y.ndim > 1):
            y = np.mean(y, axis=1)
        y[abs(y) < 1e-12] = 0.0
    except Exception as e:
        # print('expr fail')
        print(e)
        return penalty, func
    
    if (np.isnan(y).any() or np.isinf(y).any() or np.any(abs(y)>1e12)):
        print('invalid')
        return penalty, func
    
    try:
        candidate_vector = bootstrap_ela(points, y, bs_repeat=bs_repeat)
    except:
        print('ela fail')
        return penalty, func
    
    if (np.isnan(np.array(candidate_vector)).any() or np.isinf(np.array(candidate_vector)).any()):
        return penalty, func
    
    fitness = diff_vector(candidate_vector, target_vector, dist_metric=dist_metric)
    return fitness, func
# END DEF