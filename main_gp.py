

import os
import sys
import warnings
import pandas as pd
from functools import partial
from gp_fgenerator.sampling import sampling
from gp_fgenerator.gp_fgenerator import GP_func_generator
from gp_fgenerator.utils import read_pickle


#%%
def setup(pathbase, dim, ndoe, list_ela, ela_min, ela_max, fid, dist_metric='cityblock'):
    doe_x = sampling('sobol', 
                     n=ndoe,
                     lower_bound=[-5.0]*dim,
                     upper_bound=[5.0]*dim,
                     round_off=2,
                     random_seed=42,
                     verbose=True).create_doe()
    target_vector = pd.read_csv(os.path.join(pathbase, f'ela_bbob_target_f{fid}.csv'))
    
    for r in range(1):
        GP_fgen = GP_func_generator(doe_x,
                                    target_vector,
                                    minimization = True,
                                    bs_ratio = 0.8,
                                    bs_repeat = 5,
                                    list_ela = list_ela,
                                    ela_min = ela_min,
                                    ela_max = ela_max,
                                    ela_weight = {},
                                    dist_metric = dist_metric,
                                    problem_label = f'f{fid}',
                                    filepath_save = '',
                                    tree_size = (8,12),
                                    population = 20,
                                    cxpb = 0.5, 
                                    mutpb = 0.1,
                                    ngen = 5,
                                    nhof = 1,
                                    seed = r,
                                    verbose = True)   
        hof, pop = GP_fgen()
# END DEF

#%%
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    idx_nr = int(sys.argv[1])
    fid = (idx_nr % 24) +1
    dim = [2,5,10][int(idx_nr/24)]
    ndoe = 150*dim
    pathbase = os.path.join(os.getcwd(), f'results_ela_{dim}d')
    list_ela = read_pickle(os.path.join(pathbase, 'ela_bbob_corr.pickle'))
    ela_min = read_pickle(os.path.join(pathbase, 'ela_bbob_min.pickle'))
    ela_max = read_pickle(os.path.join(pathbase, 'ela_bbob_max.pickle'))
    
    setup_ = partial(setup, pathbase, dim, ndoe, list_ela, ela_min, ela_max)
    setup_(fid=fid)
# END IF