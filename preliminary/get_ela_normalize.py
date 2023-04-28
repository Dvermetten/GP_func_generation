

import os
import pandas as pd
from itertools import product
from gp_fgenerator.utils import export_pickle, dataCleaning


#%%
def get_ela_normalize(dim):
    list_fid = [i+1 for i in range(24)]
    list_iid = [i+1 for i in range(5)]
    list_bbob = list(product(list_fid, list_iid))
    df_ela = pd.DataFrame()
    for bbob in list_bbob:
        filepath = os.path.join(os.getcwd(), f'results_ela_{dim}d', f'ela_bbob_f{bbob[0]}_ins{bbob[1]}.csv')
        ela_ = pd.read_csv(filepath)
        # ela_ = ela_.mean(axis=0).to_frame().T
        df_ela = pd.concat([df_ela, ela_], axis=0, ignore_index=True)
    df_ela = dataCleaning(df_ela, replace_nan=False, inf_as_nan=True, col_allnan=False, col_anynan=False, row_anynan=False, col_null_var=False, 
                          row_dupli=False, filter_key=[], reset_index=False, verbose=False)
    dict_min = {}
    dict_max = {}
    dict_mean = {}
    dict_std = {}
    for ela in df_ela.keys():
        dict_min[ela] = df_ela[ela].min(numeric_only=True)
        dict_max[ela] = df_ela[ela].max(numeric_only=True)
        dict_mean[ela] = df_ela[ela].mean(numeric_only=True)
        dict_std[ela] = df_ela[ela].std(numeric_only=True)
    export_pickle(os.path.join(os.getcwd(), f'results_ela_{dim}d', 'ela_bbob_min.pickle'), dict_min)
    export_pickle(os.path.join(os.getcwd(), f'results_ela_{dim}d', 'ela_bbob_max.pickle'), dict_max)
    export_pickle(os.path.join(os.getcwd(), f'results_ela_{dim}d', 'ela_bbob_mean.pickle'), dict_mean)
    export_pickle(os.path.join(os.getcwd(), f'results_ela_{dim}d', 'ela_bbob_std.pickle'), dict_std)
# END DEF