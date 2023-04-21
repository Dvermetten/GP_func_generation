

import os
import warnings
import numpy as np
import pandas as pd
from functools import partial
from gp_fgenerator.sampling import sampling
from gp_fgenerator.gp_fgenerator import GP_func_generator
from gp_fgenerator.utils import runParallelFunction, read_pickle
from gp_fgenerator.visualization import plot_contour, plot_surface


#%%
def setup(pathbase, dim, ndoe, list_ela, ela_min, ela_max, fid):
    doe_x = sampling('sobol', 
                     n=ndoe,
                     lower_bound=[-5.0]*dim,
                     upper_bound=[5.0]*dim,
                     round_off=2,
                     random_seed=42,
                     verbose=True).create_doe()
    target_vector = pd.read_csv(os.path.join(pathbase, f'ela_bbob_target_f{fid}.csv'))
    
    GP_fgen = GP_func_generator(doe_x,
                                target_vector,
                                bs_ratio = 0.8,
                                bs_repeat = 5,
                                list_ela = list_ela,
                                ela_min = ela_min,
                                ela_max = ela_max,
                                ela_weight = {},
                                dist_metric = 'cityblock',
                                problem_label = f'f{fid}',
                                filepath_save = '',
                                tree_size = (8,12),
                                population = 20,
                                cxpb = 0.5, 
                                mutpb = 0.1,
                                ngen = 5,
                                nhof = 1,
                                verbose = True)   
    hof, pop = GP_fgen()
    
    # if (dim == 2):
    #     X, Y = np.meshgrid(np.arange(-5,5,0.1), np.arange(-5,5,0.1))
    #     Z = np.zeros(X.shape)
    #     for idx1 in range(100):
    #         for idx2 in range(100):
    #             Z[idx1, idx2] = np.mean(GP_fgen.fbest(np.array([[X[idx1, idx2], Y[idx1, idx2]]])), axis=1)[0]
    #     Z = (Z-Z.min())/(Z.max()-Z.min())
    #     path_dir = os.path.join(GP_fgen.filepath_save, 'plots')
    #     plot_contour(X, Y, Z, path_dir, label=f'f{fid}_gp')
    #     plot_surface(X, Y, Z, path_dir, label=f'f{fid}_3d')
# END DEF
    
#%%
def main():
    list_fid = [1] # [i+1 for i in range(24)]
    dim = 2
    ndoe = 150*dim
    np = 1
    
    pathbase = os.path.join(os.getcwd(), f'results_ela_{dim}d')
    list_ela = read_pickle(os.path.join(pathbase, 'ela_bbob_corr.pickle'))
    ela_min = read_pickle(os.path.join(pathbase, 'ela_bbob_min.pickle'))
    ela_max = read_pickle(os.path.join(pathbase, 'ela_bbob_max.pickle'))
    setup_ = partial(setup, pathbase, dim, ndoe, list_ela, ela_min, ela_max)
    runParallelFunction(setup_, list_fid, np=np)
# END DEF

#%%
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
# END IF