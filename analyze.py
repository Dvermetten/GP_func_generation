
import os
import numpy as np
import pandas as pd
from deap import gp
from functools import partial
from gp_fgenerator.utils import runParallelFunction
from gp_fgenerator.create_pset import create_pset
from gp_fgenerator.visualization import plot_contour, plot_surface



#%%
def setup(dim, fid):
    df_data = pd.read_csv(os.path.join(os.getcwd(), f'results_gpfg_{dim}d_f{fid}_seed0', 'gpfg_opt_runs.csv'))
    df_data.sort_values(by=['fitness'], ascending=True, inplace=True, ignore_index=True)
    # for i in range(5):
    i = 1
    f_ = gp.compile(df_data['strf'].iloc[i], create_pset())
    X, Y = np.meshgrid(np.arange(-5,5,0.1), np.arange(-5,5,0.1))
    Z = np.zeros(X.shape)
    for idx1 in range(100):
        for idx2 in range(100):
            Z[idx1, idx2] = np.mean(f_(np.array([X[idx1, idx2], Y[idx1, idx2]])))
    Z = (Z-Z.min())/(Z.max()-Z.min())
    path_dir = os.path.join(os.getcwd(), f'results_gpfg_{dim}d_f{fid}_seed0', 'plots')
    plot_contour(X, Y, Z, path_dir, label=f'f{fid}_top{i+1}')
    plot_surface(X, Y, Z, path_dir, label=f'f{fid}_top{i+1}')
# END DEF

#%%
def main():
    list_fid = [10] # [i+1 for i in range(1)]
    dim = 2
    np = 1
    
    setup_ = partial(setup, dim)
    runParallelFunction(setup_, list_fid, np=np)
# END DEF

#%%
if __name__ == '__main__':
    main()
# END DEF