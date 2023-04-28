

import os
import ioh
import numpy as np
import pandas as pd
from functools import partial
from gp_fgenerator.utils import runParallelFunction
from gp_fgenerator.visualization import plot_contour, plot_surface

    

#%%
def setup(dim, list_iid, path_base, fid):
    df_ela = pd.DataFrame()
    for iid in list_iid:
        filepath = os.path.join(os.getcwd(), f'results_ela_{dim}d', f'ela_bbob_f{fid}_ins{iid}.csv')
        ela_ = pd.read_csv(filepath)
        # ela_ = ela_.mean(axis=0).to_frame().T
        df_ela = pd.concat([df_ela, ela_], axis=0, ignore_index=True)
    # df_ela = df_ela.mean(axis=0).to_frame().T
    filepath = os.path.join(path_base, f'ela_bbob_target_f{fid}.csv')
    df_ela.to_csv(filepath, index=False)
    
    if (dim == 2):
        f = ioh.get_problem(fid, 1, 2)
        X, Y = np.meshgrid(np.arange(-5,5,0.1), np.arange(-5,5,0.1))
        Z = np.zeros(X.shape)
        for idx1 in range(100):
            for idx2 in range(100):
                Z[idx1, idx2] = f([X[idx1, idx2], Y[idx1, idx2]])
        Z = (Z-Z.min())/(Z.max()-Z.min())
        path_dir = os.path.join(os.getcwd(), f'results_ela_{dim}d', 'plots')
        plot_contour(X, Y, Z, path_dir, label=f'f{fid}')
        plot_surface(X, Y, Z, path_dir, label=f'f{fid}')
# END DEF

#%%
def get_ela_target(dim, np=1):
    list_fid = [i+1 for i in range(24)]
    list_iid = [i+1 for i in range(5)]
    setup_ = partial(setup, dim, list_iid, os.path.join(os.getcwd(), f'results_ela_{dim}d'))
    runParallelFunction(setup_, list_fid, np=np)
# END DEF