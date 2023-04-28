

import os
import pandas as pd
from itertools import product
from gp_fgenerator.utils import export_pickle, dataCleaning, dropFeatCorr


#%%
def get_ela_corr(dim):
    list_fid = [i+1 for i in range(24)]
    list_iid = [i+1 for i in range(5)]
    list_bbob = list(product(list_fid, list_iid))
    df_ela = pd.DataFrame()
    for bbob in list_bbob:
        filepath = os.path.join(os.getcwd(), f'results_ela_{dim}d', f'ela_bbob_f{bbob[0]}_ins{bbob[1]}.csv')
        ela_ = pd.read_csv(filepath)
        # ela_ = ela_.mean(axis=0).to_frame().T
        df_ela = pd.concat([df_ela, ela_], axis=0, ignore_index=True)
    df_clean = dataCleaning(df_ela, replace_nan=False, inf_as_nan=True, col_allnan=True, col_anynan=True, row_anynan=False, col_null_var=True, 
                            row_dupli=False, filter_key=['pca.expl_var.cov_x', 'pca.expl_var.cor_x', 'pca.expl_var_PC1.cov_x', 'pca.expl_var_PC1.cor_x'], 
                            reset_index=False, verbose=True)
    df_corr, df_pair = dropFeatCorr(df_clean, corr_thres=0.9, corr_method='pearson', mode='pair', ignore_keys=[], verbose=True)
    list_ela = list(df_corr.keys())
    export_pickle(os.path.join(os.getcwd(), f'results_ela_{dim}d', 'ela_bbob_corr.pickle'), list_ela)
# END DEF