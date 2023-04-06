
import sys
sys.path.insert(0, '/proj/cae_muc/q521100/83_Miniconda/python3.8/site-packages/')

import os
import ioh
import warnings
import numpy as np
from itertools import product
from gp_fgenerator.sampling import sampling
from gp_fgenerator.compute_ela import bootstrap_ela
from gp_fgenerator.gp_fgenerator import GP_func_generator
from gp_fgenerator.utils import plot_surface, runParallelFunction


#%%
def setup(item):
    fid = item[0]
    iid = item[1]
    dim = 2
    bs_repeat = 2
    doe_x = sampling('sobol', 
                     n=250,
                     lower_bound=[-5.0]*dim,
                     upper_bound=[5.0]*dim,
                     round_off=2,
                     random_seed=42,
                     verbose=True).create_doe()
    f = ioh.get_problem(fid, iid, dim)
    y = np.array(list(map(f, doe_x)))
    target_vector = bootstrap_ela(doe_x, y, bs_repeat=bs_repeat)
    dir_out = os.path.join(os.getcwd(), 'plots', f'F{fid}_ins{iid}')
    plot_surface(doe_x, y, dir_out, 'target')
    
    GP_fgen = GP_func_generator(doe_x,
                                target_vector,
                                dist_metric = 'canberra',
                                bs_repeat = bs_repeat,
                                problem_label = '',
                                filepath_save = '',
                                tree_size = (8,12),
                                population = 200,
                                cxpb = 0.5, 
                                mutpb = 0.1,
                                ngen = 20,
                                nhof = 1,
                                verbose = True)   
    hof, pop, logger = GP_fgen()
    ybest = GP_fgen.fbest(doe_x)
    if (ybest.ndim > 1):
        ybest = np.mean(ybest, axis=1)
    plot_surface(doe_x, ybest, dir_out, 'best')
# END DEF
    
#%%
def main():
    list_fid = [i+1 for i in range(24)]
    list_iid = [1]
    np = 6
    
    list_item = list(product(list_fid, list_iid))
    runParallelFunction(setup, list_item, np=np)
# END DEF

#%%
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
# END IF