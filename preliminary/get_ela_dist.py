

import os
import numpy as np
import pandas as pd
from functools import partial
from itertools import product, combinations
from gp_fgenerator.compute_ela import diff_vector
from gp_fgenerator.utils import read_pickle, runParallelFunction
from gp_fgenerator.visualization import plot_barplot


#%%
def setup(dim, list_ela, ela_min, ela_max, dist_metric):
    list_fid = [i+1 for i in range(24)]
    list_iid = [i+1 for i in range(5)]
    list_bbob = list(product(list_fid, list_iid))
    df_ela = pd.DataFrame()
    for bbob in list_bbob:
        filepath = os.path.join(os.getcwd(), f'results_ela_{dim}d', f'ela_bbob_f{bbob[0]}_ins{bbob[1]}.csv')
        ela_ = pd.read_csv(filepath)
        ela_ = ela_.mean(axis=0).to_frame().T
        df_ela = pd.concat([df_ela, ela_], axis=0, ignore_index=True)
    mat_dist = np.zeros((len(df_ela), len(df_ela)))
    list_pair = list(combinations(range(len(df_ela)), 2))
    for pair in list_pair:
        vec_a = df_ela.iloc[[pair[0]]]
        vec_b = df_ela.iloc[[pair[1]]]
        dist_ = diff_vector(vec_a, vec_b, list_ela=list_ela, dict_min=ela_min, dict_max=ela_max, dict_weight={}, dist_metric=dist_metric)
        mat_dist[pair[0], pair[1]] = mat_dist[pair[1], pair[0]] = dist_
    df_dist = pd.DataFrame(mat_dist, columns=list_bbob, index=list_bbob)
    df_dist.to_csv(os.path.join(os.getcwd(), f'results_ela_{dim}d', f'ela_dist_{dist_metric}.csv'), index=False)
    df_data_abs = pd.DataFrame()
    df_data_rel = pd.DataFrame()
    for fid in list_fid:
        df_avg = df_dist.loc[[(fid,i+1) for i in range(5)]]
        df_avg.drop(columns=[(fid,i+1) for i in range(5)], inplace=True)
        avg_ = df_avg.mean(axis=0).mean()
        list_abs = []
        for item in list(combinations(range(len(list_iid)), 2)):
            list_abs.append(df_dist[(fid,item[0]+1)].loc[[(fid,item[1]+1)]].item())
        list_rel = [abs/avg_ for abs in list_abs]
        df_abs = pd.DataFrame.from_dict({'dist': list_abs, 'label': f'f{fid}'})
        df_rel = pd.DataFrame.from_dict({'dist': list_rel, 'label': f'f{fid}'})
        df_data_abs = pd.concat([df_data_abs, df_abs], axis=0, ignore_index=True)
        df_data_rel = pd.concat([df_data_rel, df_rel], axis=0, ignore_index=True)
    path_dir = os.path.join(os.getcwd(), f'results_ela_{dim}d', 'plots')
    plot_barplot(df_data_abs, path_dir, label=f'{dist_metric}_abs')
    plot_barplot(df_data_rel, path_dir, label=f'{dist_metric}_rel')
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