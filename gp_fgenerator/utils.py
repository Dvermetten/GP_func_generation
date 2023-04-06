
import os
import sys
import copy
import pickle
import importlib
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from .visualization import Visualizer



#%%
def write_file(filepath, list_msg):
    with open(filepath, 'w+') as file:
        for msg in list_msg:
            file.write(msg + '\n')
# END DEF

#%%
def write_af(path_output, label_af, str_af):
    filename = f"af_{label_af}"
    filepath = os.path.join(path_output, f"{filename}.py")
    list_msg = ['import numpy as np\n',
                f'def {filename}(array_x):',
                f'    return {str_af}']
    write_file(filepath, list_msg)
# END DEF

#%%
def eval_af(path_output, label_af):
    filename = f"af_{label_af}"
    filepath = os.path.join(path_output, f"{filename}.py")
    if not (os.path.isfile(filepath)):
        raise ValueError(f'File {filepath} is missing.')
    sys.path.append(path_output)
    module_ = importlib.import_module(filename)
    importlib.reload(module_)
    func_ = getattr(module_, filename)
    return func_
# END DEF

#%%
# export pickle
def export_pickle(filepath, data):
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# END DEF

#%%
# read pickle
def read_pickle(filepath):
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    return data
# END DEF

#%%
def dataCleaning(df_data,
                 replace_nan=None,
                 inf_as_nan=False,
                 col_allnan=False, 
                 col_anynan=False, 
                 row_anynan=False, 
                 col_null_var=True,
                 row_dupli=True, 
                 filter_key=[], 
                 reset_index=True,
                 verbose=True):
    df_data_clean = copy.deepcopy(df_data)
    # replace all inf as nan
    if (inf_as_nan):
        df_data_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        if (verbose):
            print('All -inf and inf have been replaced with NaN.\n')
            
    # replace all nan with a value
    if (replace_nan):
        df_data_clean.fillna(replace_nan)
        if (verbose):
            print(f'All missing values have been replaced with {replace_nan}.\n')
            
    # drop columns with key of a specific string    
    if (filter_key):
        for key in filter_key:
            df_data_clean = df_data_clean[df_data_clean.columns.drop(list(df_data_clean.filter(regex=key)))]
        list_filter_key = []
        for key in df_data.keys():
            if (key not in df_data_clean.keys()):
                list_filter_key.append(key)
        if (list_filter_key and verbose):
            print(f'Following {len(list_filter_key)} features with filter keyword have been dropped: {list_filter_key}.\n')
    
    # drop columns with all missing value or NAN
    if (col_allnan):
        col_to_drop = []
        for col in df_data_clean.columns:
            df_temp = df_data_clean[[col]]
            if (df_temp.isnull().values.all()):
                col_to_drop.append(col)
        if (col_to_drop):
            df_data_clean.drop(col_to_drop, axis=1, inplace=True)
            if (verbose):
                print(f'Following {len(col_to_drop)} columns with ONLY NAN have been dropped: {col_to_drop}.\n')
    
    # drop columns with any missing value or NAN
    if (col_anynan):
        col_to_drop = []
        for col in df_data_clean.columns:
            df_temp = df_data_clean[[col]]
            if (df_temp.isnull().values.any()):
                col_to_drop.append(col)
        if (col_to_drop):
            df_data_clean.drop(col_to_drop, axis=1, inplace=True)
            if (verbose):
                print(f'Following {len(col_to_drop)} columns with ANY NAN have been dropped: {col_to_drop}.\n')
    
    # drop row with any missing value or NAN
    if (row_anynan):
        row_to_drop = []
        for ind in range(len(df_data_clean)):
            df_temp = df_data_clean.iloc[ind]
            
            if (df_temp.isnull().values.any()):
                row_to_drop.append(df_data_clean.index[ind])
        if (row_to_drop):
            df_data_clean.drop(row_to_drop, axis=0, inplace=True)
            if (verbose):
                print(f'Following {len(row_to_drop)} rows with ANY NAN have been dropped: {row_to_drop}.\n')
    
    # drop columns with no variance or 0 variance
    if (col_null_var):
        counts = df_data_clean.nunique()
        to_del = [counts.index[i] for i,v in enumerate(counts) if v == 1]
        if (to_del):
            df_data_clean.drop(to_del, axis=1, inplace=True)
            if (verbose):
                print(f'Following {len(to_del)} features with 0 variance have been dropped: {to_del}.\n')
    
    # drop duplicated rows
    if (row_dupli):
        dups = df_data_clean.duplicated()
        if (dups.any()):
            df_data_clean.drop_duplicates(inplace=True)
            if (verbose):
                print('Duplicated rows have been dropped.\n')

    if (reset_index):
        df_data_clean.reset_index(drop=True, inplace=True)
        if (verbose):
            print('Index of dataframe has been reset.\n')
    if (verbose):
        print(f'Final {len(df_data_clean.keys())} features {list(df_data_clean.keys())} remain.')
    return df_data_clean
# END DEF

#%%
def plot_surface(xdoe, ydoe, dir_out, label):
    if not (os.path.isdir(dir_out)):
        os.makedirs(dir_out)
    visu = Visualizer()
    x_data = pd.DataFrame(xdoe, columns=['x', 'y'])
    y_data = pd.DataFrame(ydoe, columns=['z'])
    df_plot = pd.concat([x_data, y_data], axis=1)
    dict_plot = {'df_data': df_plot,
                 'x_name': 'x',
                 'y_name': 'y',
                 'z_name': 'z',
                 'surface3d': True,
                 'scatter3d': False,
                 'alpha_surf': 1.0,
                 'angle': (45,-135),
                 'xbound': (-4,4),
                 'ybound': (-4,4),
                 'cbar_label': 'f'}
    figname = f'plot_3d_{label}'
    visu.plotSingle3D('scatter3Dplot', title='', dir_out=dir_out, figname=figname, figformat='.png', figsize=(6,3), dpi=300, show=False,
                        xlim=(None,None), ylim=(None,None), zlim=(None,None), 
                        xlabel='x', ylabel='y', zlabel='', **dict_plot)
# END DEF

#%%
def runParallelFunction(runFunction, arguments, np=1):
    p = Pool(min(cpu_count(), len(arguments), np))
    results = p.map(runFunction, arguments)
    p.close()
    return results
# END DEF