
import os
import ioh
import numpy as np
from itertools import product
from functools import partial
from gp_fgenerator.sampling import sampling
from gp_fgenerator.compute_ela import bootstrap_ela
from gp_fgenerator.utils import runParallelFunction



#%%
def setup(path_base, X, bbob):
    fid = bbob[0]
    iid = bbob[1]
    f = ioh.get_problem(fid, iid, X.shape[1])
    y = np.array(list(map(f, X)))
    df_ela = bootstrap_ela(X, y, bs_ratio=0.8, bs_repeat=5)
    filepath = os.path.join(path_base, f'ela_bbob_f{fid}_ins{iid}.csv')
    df_ela.to_csv(filepath, index=False)
    print(f'ELA BBOB F{fid} ins{iid} done.')
# END DEF

#%%
def get_ela_bbob(dim, ndoe, np=1):
    list_fid = [i+1 for i in range(24)]
    list_iid = [i+1 for i in range(5)]
    X = sampling('sobol',
                 n=ndoe,
                 lower_bound=[-5.0]*dim,
                 upper_bound=[5.0]*dim,
                 round_off=2,
                 random_seed=42,
                 verbose=True).create_doe()
    list_bbob = list(product(list_fid, list_iid))
    path_base = os.path.join(os.getcwd(), f'results_ela_{dim}d')
    if not (os.path.isdir(path_base)):
        os.makedirs(path_base)
    setup_ = partial(setup, path_base, X)
    runParallelFunction(setup_, list_bbob, np=np)
# END DEF