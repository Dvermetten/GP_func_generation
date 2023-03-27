import numpy as np
import pandas as pd
from functools import partial
import ioh
import warnings 

from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample

from scipy import spatial

maxs = np.load('maxs.npy')
mins = np.load('mins.npy')

def get_ela_list(problem):
   # Create sample
    X = create_initial_sample(problem.meta_data.n_variables, lower_bound = -5, upper_bound = 5)
    y = X.apply(lambda x: problem(x), axis = 1)
    y = (max(y) - y) / (max(y)-min(y))
    # Calculate ELA features
    ela_meta = calculate_ela_meta(X, y)
    ela_distr = calculate_ela_distribution(X, y)
    ela_level = calculate_ela_level(X, y)
    nbc = calculate_nbc(X, y)
    disp = calculate_dispersion(X, y)
    ic = calculate_information_content(X, y, seed = 100)

    # Concatenate results in list
    ela = list(ic.values())[:-1]
    [ela.append(x) for x in list(ela_meta.values())[:-1]]
    [ela.append(x) for x in list(ela_distr.values())[:-1]]
    [ela.append(x) for x in list(ela_level.values())[:-1]]
    [ela.append(x) for x in list(nbc.values())[:-1]]
    [ela.append(x) for x in list(disp.values())[:-1]]
    return np.array(ela)


def obj_func_internal(problem, target_vector, distance_measure):
    vector = get_ela_list(problem)
    a_norm = maxs-target_vector/(maxs-mins)
    b_norm = maxs-vector/(maxs-mins)
    return distance_measure(a_norm,b_norm)

def main():
    target = get_ela_list(ioh.get_problem(1,1,5))
    obj_func = partial(obj_func_internal, target_vector = target, distance_measure=spatial.distance.cosine)
    
    print(obj_func(ioh.get_problem(1,1,5)))
    print(obj_func(ioh.get_problem(21,1,5)))

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()