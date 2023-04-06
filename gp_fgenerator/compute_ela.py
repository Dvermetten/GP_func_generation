
import math
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.utils import resample
import pflacco.classical_ela_features as pflacco_ela
from gp_fgenerator.utils import read_pickle, dataCleaning




#%%
def compute_ela(X, y, lower_bound=-5, upper_bound=5):
    y_rescale = (max(y) - y) / (max(y)-min(y))
    # Calculate ELA features
    ela_meta = pflacco_ela.calculate_ela_meta(X, y_rescale)
    ela_distr = pflacco_ela.calculate_ela_distribution(X, y_rescale)
    ela_level = pflacco_ela.calculate_ela_level(X, y_rescale)
    pca = pflacco_ela.calculate_pca(X, y_rescale)
    limo = pflacco_ela.calculate_limo(X, y_rescale, lower_bound, upper_bound)
    nbc = pflacco_ela.calculate_nbc(X, y_rescale)
    disp = pflacco_ela.calculate_dispersion(X, y_rescale)
    ic = pflacco_ela.calculate_information_content(X, y_rescale, seed=100)
    ela_ = {**ela_meta, **ela_distr, **ela_level, **pca, **limo, **nbc, **disp, **ic}
    df_ela = pd.DataFrame([ela_])
    return df_ela
# END DEF

#%%
def bootstrap_ela(X, y, lower_bound=-5, upper_bound=5, bs_size=0.8, bs_repeat=2, bs_seed=42):
    num_sample = int(math.ceil(len(X) * bs_size))
    df_ela = pd.DataFrame()
    for i_bs in range(bs_repeat):
        i_bs_seed = i_bs + bs_seed
        X_, y_ = resample(X, y, replace=False, n_samples=num_sample, random_state=i_bs_seed, stratify=None)
        ela_ = compute_ela(X_, y_, lower_bound=lower_bound, upper_bound=upper_bound)
        df_ela = pd.concat([df_ela, ela_], axis=0, ignore_index=True)
    return df_ela.mean(axis=0).to_frame().T
# END DEF

#%%
def clean_ela(candidate_ela, target_ela):
    df_ela_bbob = read_pickle('/proj/cae_muc/q521100/82_Python_Workspace/2023/230327_gp_fgenerator/ela_bbob_2d.pickle')
    df_ela_bbob.drop(columns=['fid', 'iid'], inplace=True)
    df_clean = dataCleaning(df_ela_bbob, replace_nan=False, inf_as_nan=True, col_allnan=True, col_anynan=True, row_anynan=False, col_null_var=True, 
                            row_dupli=True, filter_key=['.costs_runtime'], reset_index=True, verbose=False)
    maxs = df_clean.max(axis=0).to_frame().T
    mins = df_clean.min(axis=0).to_frame().T
    df_ela = pd.concat([maxs, mins, candidate_ela, target_ela], axis=0, join='inner', ignore_index=True)
    return np.array(df_ela.iloc[0]), np.array(df_ela.iloc[1]), np.array(df_ela.iloc[2]), np.array(df_ela.iloc[3])
# END DEF    

#%%
def diff_vector(candidate_vector, target_vector, dist_metric):
    maxs, mins, cand_, targ_ = clean_ela(candidate_vector, target_vector)
    a_norm = maxs-targ_/(maxs-mins)
    b_norm = maxs-cand_/(maxs-mins)
    dict_dist_metric = {'canberra': spatial.distance.canberra,
                        'cosine': spatial.distance.cosine,
                        'correlation': spatial.distance.correlation,
                        'euclidean': spatial.distance.euclidean,
                        'cityblock': spatial.distance.cityblock}
    return dict_dist_metric[dist_metric](a_norm,b_norm)
# END DEF