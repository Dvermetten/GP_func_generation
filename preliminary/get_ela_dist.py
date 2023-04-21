

import os
import pandas as pd
from functools import partial
from itertools import combinations
from gp_fgenerator.compute_ela import diff_vector
from gp_fgenerator.utils import read_pickle, runParallelFunction
from gp_fgenerator.visualization import plot_barplot


#%%
def setup(dim, list_ela, ela_min, ela_max, dist_metric):
    list_fid = [i+1 for i in range(24)]
    list_iid = [i+1 for i in range(5)]
    df_data = pd.DataFrame()
    for fid in list_fid:
        df_ela = pd.DataFrame()
        for iid in list_iid:
            filepath = os.path.join(os.getcwd(), f'results_ela_{dim}d', f'ela_bbob_f{fid}_ins{iid}.csv')
            ela_ = pd.read_csv(filepath)
            ela_ = ela_.mean(axis=0).to_frame().T
            df_ela = pd.concat([df_ela, ela_], axis=0, ignore_index=True)
        list_dist = []
        list_pair = list(combinations(range(len(df_ela)), 2))
        for pair in list_pair:
            vec_a = df_ela.iloc[[pair[0]]]
            vec_b = df_ela.iloc[[pair[1]]]
            dist_ = diff_vector(vec_a, vec_b, list_ela=list_ela, dict_min=ela_min, dict_max=ela_max, dict_weight={}, dist_metric=dist_metric)
            list_dist.append(dist_)
        df_ = pd.DataFrame.from_dict({'dist': list_dist, 'label': f'f{fid}'})
        df_data = pd.concat([df_data, df_], axis=0, ignore_index=True)
    path_dir = os.path.join(os.getcwd(), f'results_ela_{dim}d', 'plots')
    plot_barplot(df_data, path_dir, label=dist_metric)
# END DEF

#%%
def get_ela_dist(dim, np=1):
    list_dist = ['cityblock', 'cosine', 'correlation', 'canberra', 'euclidean']
    pathbase = os.path.join(os.getcwd(), f'results_ela_{dim}d')
    list_ela = read_pickle(os.path.join(pathbase, 'ela_bbob_corr.pickle'))
    ela_min = read_pickle(os.path.join(pathbase, 'ela_bbob_min.pickle'))
    ela_max = read_pickle(os.path.join(pathbase, 'ela_bbob_max.pickle'))
    setup_ = partial(setup, dim, list_ela, ela_min, ela_max)
    runParallelFunction(setup_, list_dist, np=np)
# END DEF