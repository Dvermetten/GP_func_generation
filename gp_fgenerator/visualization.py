# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:43:23 2022

@author: Q521100
"""


import os
import math
import copy
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import scipy.interpolate
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import colors
from scipy.stats import pearsonr



__authors__ = ["Fu Xing Long"]



#%%
##################################
'''
# Utility
'''
##################################

#%%
def convert2rank(df_base, x_name, y_name, hue=None, ascending=True):
    df = df_base.sort_values(by=[x_name, y_name], ascending=ascending)
    df_plot = pd.DataFrame()
    for item in list(df[x_name].unique()):
        df_temp = copy.deepcopy(df[df[x_name] == item])
        df_temp['rank'] = [i+1 for i in range(len(df_temp))]
        df_plot = pd.concat([df_plot, df_temp], axis=0, ignore_index=True)
        
    if (hue):
        df_rank = pd.DataFrame()
        list_hue = list(df_base[hue].unique())
        for item in list_hue:
            df_temp = df_plot[df_plot[hue] == item]
            df_temp.sort_values(by=[x_name], ascending=True, inplace=True)
            df_rank = pd.concat([df_rank, df_temp], axis=0, ignore_index=True)
    else:
        df_rank = df_plot
    return df_rank
# END DEF
                
                
                
#%%
##################################
'''
# Single Plot Function
'''
##################################

#%%
# line plot
def lineplot(**dict_plot):
    # parameters
    data = dict_plot['df_data']
    x_name = dict_plot['x_name']
    y_name = dict_plot['y_name']
    # optional
    hue = dict_plot['hue'] if 'hue' in dict_plot.keys() else None
    list_hue = dict_plot['list_hue'] if 'list_hue' in dict_plot.keys() else []
    style = dict_plot['style'] if 'style' in dict_plot.keys() else None
    lw = dict_plot['lw'] if 'lw' in dict_plot.keys() else 1.
    df_fill = dict_plot['df_fill'] if 'df_fill' in dict_plot.keys() else pd.DataFrame()
    alpha_fill = dict_plot['alpha_fill'] if 'alpha_fill' in dict_plot.keys() else 0.5
    color_fill = dict_plot['color_fill'] if 'color_fill' in dict_plot.keys() else sns.color_palette("tab10")
    scatter = dict_plot['scatter'] if 'scatter' in dict_plot.keys() else False
    palette = dict_plot['palette'] if 'palette' in dict_plot.keys() else None
    # plot
    ax = dict_plot['ax']
    if (hue):
        df_plot = pd.DataFrame()
        if (list_hue):
            list_hue_temp = list_hue
        else:
            list_hue_temp = list(data[hue].unique())
        for item in list_hue_temp:
            df_temp = copy.deepcopy(data[data[hue] == item])
            df_temp.sort_values(by=[x_name], ascending=True, inplace=True)
            df_plot = pd.concat([df_plot, df_temp], axis=0, ignore_index=True)
    else:
        df_plot = copy.deepcopy(data)
    sns.lineplot(data=df_plot, x=x_name, y=y_name, hue=hue, style=style, lw=lw, ax=ax, palette=palette)
    if (scatter):
        sns.scatterplot(data=df_plot, x=x_name, y=y_name, hue=hue, style=style, s=10, ax=ax, palette=palette)
    if not (df_fill.empty):
        for i, item in enumerate(list(df_fill[hue].unique())):
            df_fill_temp = df_fill[df_fill[hue] == item]
            array_x = np.array(df_fill_temp[x_name])
            array_ymean = np.array(df_fill_temp['mean'])
            array_ylower = np.array(df_fill_temp['lower'])
            array_yupper = np.array(df_fill_temp['upper'])
            ax.plot(array_x, array_ymean, color=color_fill[i])
            if (scatter):
                ax.scatter(array_x, array_ymean, color=color_fill[i], s=10, marker='x')
            ax.fill_between(array_x, array_ylower, array_yupper, alpha=alpha_fill, color=color_fill[i])
# END DEF

#%%
# scatter plot and/or 2D contour plot
def scatterplot(**dict_plot):
    # parameters
    data = dict_plot['df_data']
    x_name = dict_plot['x_name']
    y_name = dict_plot['y_name']
    # optional
    # contour
    contour = dict_plot['contour'] if 'contour' in dict_plot.keys() else False
    z_name = dict_plot['z_name'] if 'z_name' in dict_plot.keys() else ''
    xbound = dict_plot['xbound'] if 'xbound' in dict_plot.keys() else None
    ybound = dict_plot['ybound'] if 'ybound' in dict_plot.keys() else None
    fill = dict_plot['fill'] if 'fill' in dict_plot.keys() else True
    cbar = dict_plot['cbar'] if 'cbar' in dict_plot.keys() else None
    cbar_label = dict_plot['cbar_label'] if 'cbar_label' in dict_plot.keys() else ''
    interp = dict_plot['interp'] if 'interp' in dict_plot.keys() else 'cubic'
    cmap = dict_plot['cmap'] if 'cmap' in dict_plot.keys() else cm.jet
    # scatter
    scatter = dict_plot['scatter'] if 'scatter' in dict_plot.keys() else True
    hue = dict_plot['hue'] if 'hue' in dict_plot.keys() else None
    style = dict_plot['style'] if 'style' in dict_plot.keys() else None
    palette = dict_plot['palette'] if 'palette' in dict_plot.keys() else None
    scatter_size = dict_plot['scatter_size'] if 'scatter_size' in dict_plot.keys() else 5
    label_text = dict_plot['label_text'] if 'label_text' in dict_plot.keys() else ''
    text_offset = dict_plot['text_offset'] if 'text_offset' in dict_plot.keys() else []
    markers = dict_plot['markers'] if 'markers' in dict_plot.keys() else True
    # plot
    ax = dict_plot['ax']
    if not (contour or scatter):
        raise ValueError('Either scatter and/or contour plot must be activated.')
    if (contour):
        df_xy = data[[x_name, y_name]].to_numpy()
        df_z = data[[z_name]]
        if cbar:
            vmin = cbar[0]
            vmax = cbar[1]
        else:
            vmin = df_z.min()
            vmax = df_z.max()
        list_z = []
        for i in range(len(df_z)):
            list_z.append(df_z[z_name].iloc[i])
        array_z = np.array(list_z)
        if (xbound):
            xi = np.linspace(xbound[0], xbound[1], len(data))
        else:
            xi = np.linspace(data[x_name].min(), data[x_name].max(), len(data))
        if (ybound):
            yi = np.linspace(ybound[0], ybound[1], len(data))
        else:
            yi = np.linspace(data[y_name].min(), data[y_name].max(), len(data))
        zi = scipy.interpolate.griddata(df_xy, array_z, (xi[None,:], yi[:,None]), method=interp)
        if (fill):
            plt.contourf(xi, yi, zi, cmap=cmap, vmax=vmax, vmin=vmin)
        else:
            plt.contour(xi, yi, zi, cmap=cmap, vmax=vmax, vmin=vmin)
        cb = plt.colorbar()
        cb.set_label(cbar_label)
    if (scatter):
        sns.scatterplot(data=data, x=x_name, y=y_name, hue=hue, style=style, palette=palette, s=scatter_size, markers=markers, ax=ax)
        # label text
        if (label_text):
            for i, point in data.iterrows():
                if (text_offset):
                    x_offset = text_offset[i][0]
                    y_offset = text_offset[i][-1]
                    ax.text(point[x_name]+x_offset, point[y_name]+y_offset, str(point[label_text]))
                else:
                    ax.text(point[x_name]+.02, point[y_name], str(point[label_text]))
# END DEF

#%%
# dendrogram
def dendrogramplot(**dict_plot):
    # parameters
    linkage_matrix = dict_plot['linkage_matrix']
    # optional
    label_angle = dict_plot['label_angle'] if 'label_angle' in dict_plot.keys() else None
    label_ha = dict_plot['label_ha'] if 'label_ha' in dict_plot.keys() else 'center'
    mark_hline = dict_plot['mark_hline'] if 'mark_hline' in dict_plot.keys() else []
    cluster_color_thres = dict_plot['cluster_color_thres'] if 'cluster_color_thres' in dict_plot.keys() else None
    tick_hl = dict_plot['tick_hl'] if 'tick_hl' in dict_plot.keys() else {}
    hl_pad = dict_plot['hl_pad'] if 'hl_pad' in dict_plot.keys() else 0.2
    hl_alpha = dict_plot['hl_alpha'] if 'hl_alpha' in dict_plot.keys() else 0.3
    kwargs = dict_plot['kwargs'] if 'kwargs' in dict_plot.keys() else {}
    # plot
    if (mark_hline):
        for thres_line in mark_hline:
            plt.axhline(y=thres_line, color='k', linestyle='--')
    R = dendrogram(linkage_matrix, color_threshold=cluster_color_thres, **kwargs)
    ax = dict_plot['ax']
    if (label_angle):
        plt.setp(ax.get_xticklabels(), rotation=label_angle, horizontalalignment=label_ha)
    if (tick_hl):
        for key_type in tick_hl.keys():
            for color in tick_hl[key_type].keys():
                list_label_temp = tick_hl[key_type][color]
                for i in range(len(ax.get_xticklabels())):
                    key_label = ax.get_xticklabels()[i].get_text()
                    for label_temp in list_label_temp:
                        if (key_type == 'implicit'):
                            if (label_temp in key_label):
                                ax.get_xticklabels()[i].set_bbox(dict(boxstyle='round', pad=hl_pad, fc=color, alpha=hl_alpha))
                        elif (key_type == 'explicit'):
                            if (label_temp == key_label):
                                ax.get_xticklabels()[i].set_bbox(dict(boxstyle='round', pad=hl_pad, fc=color, alpha=hl_alpha))
    ax.tick_params(axis='x', which='major', labelsize=dict_plot['font_size'])
    return R
# END DEF

#%%
# heatmap plot
def heatmapplot(**dict_plot):
    # parameters
    data = dict_plot['df_data']
    # optional
    vmin = dict_plot['vmin'] if 'vmin' in dict_plot.keys() else None
    vmax = dict_plot['vmax'] if 'vmax' in dict_plot.keys() else None
    mask = dict_plot['mask'] if 'mask' in dict_plot.keys() else None
    xticklabels = dict_plot['xticklabels'] if 'xticklabels' in dict_plot.keys() else 'auto'
    yticklabels = dict_plot['yticklabels'] if 'yticklabels' in dict_plot.keys() else 'auto'
    annot = dict_plot['annot'] if 'annot' in dict_plot.keys() else False
    cmap = dict_plot['cmap'] if 'cmap' in dict_plot.keys() else sns.diverging_palette(240,10,as_cmap=True)
    # plot
    sns.heatmap(data, vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels, cmap=cmap, mask=mask, annot=annot)
    ax = dict_plot['ax']
    plt.setp(ax.get_yticklabels(), rotation=360, horizontalalignment='right')
# END DEF

#%%
# box plot
def boxplot(**dict_plot):
    # parameters
    data = dict_plot['df_data']
    x_name = dict_plot['x_name']
    y_name = dict_plot['y_name']
    # optional
    hue = dict_plot['hue'] if 'hue' in dict_plot.keys() else None
    orient = dict_plot['orient'] if 'orient' in dict_plot.keys() else 'v'
    scatter = dict_plot['scatter'] if 'scatter' in dict_plot.keys() else False
    showfliers = dict_plot['showfliers'] if 'showfliers' in dict_plot.keys() else True
    fliersize = dict_plot['fliersize'] if 'fliersize' in dict_plot.keys() else 5
    linewidth = dict_plot['linewidth'] if 'linewidth' in dict_plot.keys() else None
    # plot
    ax = dict_plot['ax']
    sns.boxplot(data=data, x=x_name, y=y_name, hue=hue, orient=orient, showfliers=showfliers, fliersize=fliersize, linewidth=linewidth, ax=ax)
    if (scatter):
        sns.swarmplot(data=data, x=x_name, y=y_name, hue=hue, color=".25", ax=ax)
# END DEF

#%%
# bar plot
def barplot(**dict_plot):
    # parameters
    data = dict_plot['df_data']
    x_name = dict_plot['x_name']
    y_name = dict_plot['y_name']
    # optional
    hue = dict_plot['hue'] if 'hue' in dict_plot.keys() else None
    orient = dict_plot['orient'] if 'orient' in dict_plot.keys() else 'v'
    rot_angle = dict_plot['rot_angle'] if 'rot_angle' in dict_plot.keys() else None
    # plot
    ax = dict_plot['ax']
    sns.barplot(data=data, x=x_name, y=y_name, hue=hue, orient=orient, ax=ax)
    if (rot_angle):
        plt.setp(ax.get_xticklabels(), rotation=rot_angle, horizontalalignment='center')
# END DEF

#%%
# categorical plot
# TODO: not working
def catplot(**dict_plot):
    # parameters
    data = dict_plot['df_data']
    x_name = dict_plot['x_name']
    y_name = dict_plot['y_name']
    kind = dict_plot['kind'] # 'strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count'
    col = dict_plot['col']
    # optional
    hue = dict_plot['hue'] if 'hue' in dict_plot.keys() else None
    orient = dict_plot['orient'] if 'orient' in dict_plot.keys() else 'v'
    # plot
    ax = dict_plot['ax']
    sns.catplot(data=data, x=x_name, y=y_name, hue=hue, kind=kind, col=col, orient=orient, ax=ax)
# END DEF
    
            
#%%
# correlogram plot
def correlogramplot(**dict_plot):
    # parameters
    data = dict_plot['df_data']
    # optional
    regr = dict_plot['regr'] if 'regr' in dict_plot.keys() else False
    regr_color = dict_plot['regr_color'] if 'regr_color' in dict_plot.keys() else 'k'
    regr_lw = dict_plot['regr_lw'] if 'regr_lw' in dict_plot.keys() else 1.
    size_scatter = dict_plot['size_scatter'] if 'size_scatter' in dict_plot.keys() else 5
    regr_ci = dict_plot['regr_ci'] if 'regr_ci' in dict_plot.keys() else None
    
    def pearson_coef(x, y, corr='pearson', corr_pos=(0.5,0.5), **kwargs):
        ax = plt.gca()
        cmap = cm.get_cmap('coolwarm_r')
        norm = colors.Normalize(vmin=-1.0, vmax=1.0)
        if (corr == 'pearson'):
            r,p = pearsonr(x,y)
        ax.set_facecolor(cmap(norm(r)))
        ax.annotate(r'$\rho$ = {:.2f}'.format(r), xy=corr_pos, xycoords='axes fraction', ha='center',
                    bbox=dict(boxstyle='round,pad=0.9', fc=cmap(norm(r)), alpha=1.0))
        ax.set_axis_off()
    # plot   
    g = sns.PairGrid(data)
    g.map_diag(sns.histplot, kde=True)
    if (regr):
        g.map_lower(sns.regplot, line_kws={'color': regr_color, 'linewidth': regr_lw}, scatter_kws={'s': size_scatter}, ci=regr_ci)
    else:
        g.map_lower(plt.scatter, s=size_scatter)
    g.map_upper(pearson_coef)
    
    # diagramm properties
    title = dict_plot['title']
    dir_out = dict_plot['dir_out']
    figname = dict_plot['figname']
    figformat = dict_plot['figformat']
    figsize = dict_plot['figsize']
    dpi = dict_plot['dpi']
    show = dict_plot['show']
    # set properties
    g.fig.set_figheight(figsize[1])
    g.fig.set_figwidth(figsize[0])
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title)       
    if (figname and dir_out):
        figname = figname + figformat
        filepath = os.path.join(dir_out, figname)
        g.savefig(filepath, bbox_inches='tight', dpi=dpi)
    if (show):
        plt.show()
    else:
        plt.close()
# END DEF        

#%%
# violin plot
def violinplot(**dict_plot):
    # parameters
    data = dict_plot['df_data']
    # optional
    fig_col = dict_plot['fig_col'] if 'fig_col' in dict_plot.keys() else 2
    violin_color = dict_plot['violin_color'] if 'violin_color' in dict_plot.keys() else []
    inner = dict_plot['inner'] if 'inner' in dict_plot.keys() else 'quartile'
    scatter_type = dict_plot['scatter_type'] if 'scatter_type' in dict_plot.keys() else 'swarm'
    scatter_plot = dict_plot['scatter_plot'] if 'scatter_plot' in dict_plot.keys() else False
    scatter_group = dict_plot['scatter_group'] if 'scatter_group' in dict_plot.keys() else None
    scatter_hue = dict_plot['scatter_hue'] if 'scatter_hue' in dict_plot.keys() else None
    scatter_size = dict_plot['scatter_size'] if 'scatter_size' in dict_plot.keys() else 5
    scatter_marker = dict_plot['scatter_marker'] if 'scatter_marker' in dict_plot.keys() else ['x', '^', 'o']
    scatter_color = dict_plot['scatter_color'] if 'scatter_color' in dict_plot.keys() else ['r', 'b', 'k']
    scatter_jitter = dict_plot['strip_jitter'] if 'strip_jitter' in dict_plot.keys() else True
    rot_angle = dict_plot['rot_angle'] if 'rot_angle' in dict_plot.keys() else None
    ax_color = dict_plot['ax_color'] if 'ax_color' in dict_plot.keys() else []
    
    # plot
    figsize = dict_plot['figsize']
    dpi = dict_plot['dpi']
    list_feature = list(data.keys())
    if (scatter_type not in ['strip', 'swarm']):
        raise ValueError(f'Scatter type {scatter_type} is not defined.')
    if (scatter_hue):
        list_feature.remove(scatter_hue)
    if (scatter_group):
        list_feature.remove(scatter_group)
    if (fig_col >= 2):
        fig_row = math.ceil(len(list_feature) / fig_col)
        fig, axs = plt.subplots(nrows=fig_row, ncols=fig_col, figsize=figsize, dpi=dpi, sharex=False)
        for i, ax in enumerate(axs.flat):
            if (i >= len(list_feature)):
                ax.set_axis_off()
                continue
            ax.set_axisbelow(True)
            ax.grid(b=True,lw=.5)
            
            df_temp = copy.deepcopy(data)
            df_temp.index.name = 'label'
            df_temp.reset_index(drop=False, inplace=True)
            df_temp.rename({'label': 'x', list_feature[i]: 'y'}, axis=1, inplace=True)
            if (scatter_hue):
                df_temp[scatter_hue] = data[[scatter_hue]]
            if (violin_color):
                color = violin_color[i]
            else:
                color = '0.8'
            sns.violinplot(x='x', y='y', data=df_temp, color=color, inner=inner, ax=ax)
            if (scatter_plot):
                if (scatter_group):
                    list_group = list(df_temp[scatter_group].unique())
                    for group, marker, color, size in zip(list_group, scatter_marker, scatter_color, scatter_size):
                        df_group_temp = copy.deepcopy(df_temp[df_temp[scatter_group] == group])
                        df_group_temp.drop([scatter_group], axis=1, inplace=True)
                        if (scatter_type=='strip'):
                            sns.stripplot(x='x', y='y', data=df_group_temp, marker=marker, color=color, linewidth=size/2, jitter=scatter_jitter, 
                                          zorder=1, size=size, ax=ax)
                        elif (scatter_type=='swarm'):
                            sns.swarmplot(x='x', y='y', data=df_group_temp, marker=marker, color=color, linewidth=size/2, size=size, ax=ax)
                else:
                    size = scatter_size[0]
                    if (scatter_type=='strip'):
                        sns.stripplot(x='x', y='y', data=df_temp, hue=scatter_hue, marker='o', palette=['k'], linewidth=size/2, jitter=scatter_jitter, 
                                      zorder=1, size=size, ax=ax)
                    elif (scatter_type=='swarm'):
                        sns.swarmplot(x='x', y='y', data=df_temp, hue=scatter_hue, marker='o', palette=['k'], linewidth=size/2, size=size, ax=ax)
            ax.set(xlabel=None)
            ax.set(ylabel=None)
            if (rot_angle):
                plt.setp(ax.get_xticklabels(), rotation=rot_angle, horizontalalignment='center')
            ax.set_title(list_feature[i])
            if (ax_color):
                if (ax_color[i]):
                    ax.set_facecolor(ax_color[i])
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_axisbelow(True)
        ax.grid(b=True,lw=.5)
        
        df_temp = copy.deepcopy(data)
        df_temp.index.name = 'label'
        df_temp.reset_index(drop=False, inplace=True)
        df_temp.rename({'label': 'x', list_feature[0]: 'y'}, axis=1, inplace=True)
        if (scatter_hue):
            df_temp[scatter_hue] = data[[scatter_hue]]
        if (violin_color):
            color = violin_color[0]
        else:
            color = '0.8'
        sns.violinplot(x='x', y='y', data=df_temp, color=color, inner=inner, ax=ax)
        if (scatter_plot):
            size = scatter_size[0]
            if (scatter_type=='strip'):
                sns.stripplot(x='x', y='y', data=df_temp, hue=scatter_hue, marker='o', palette=['k'], linewidth=size/2, jitter=scatter_jitter, 
                              zorder=1, size=size, ax=ax)
            elif (scatter_type=='swarm'):
                sns.swarmplot(x='x', y='y', data=df_temp, hue=scatter_hue, marker='o', palette=['k'], linewidth=size/2, size=size, ax=ax)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        if (rot_angle):
            plt.setp(ax.get_xticklabels(), rotation=rot_angle, horizontalalignment='center')
        ax.set_title(list_feature[0])
        if (ax_color):
            if (ax_color[0]):
                ax.set_facecolor(ax_color[0])
    
    # diagramm properties
    title = dict_plot['title']
    dir_out = dict_plot['dir_out']
    figname = dict_plot['figname']
    figformat = dict_plot['figformat']
    show = dict_plot['show']
    # set properties
    fig.suptitle(title)
    fig.tight_layout()
    if (figname and dir_out):
        figname = figname + figformat
        filepath = os.path.join(dir_out, figname)
        plt.savefig(filepath, bbox_inches='tight')
    if (show):
        plt.show()
    else:
        plt.close(fig)
# END DEF



#%%
##################################
'''
# 3D Plot Function
'''
##################################

#%%
# scatter 3D plot and/or surface 3D plot with contour 2D plot and scatter 2D plot
def scatter3Dplot(**dict_plot):
    # parameters
    data = dict_plot['df_data']
    x_name = dict_plot['x_name']
    y_name = dict_plot['y_name']
    z_name = dict_plot['z_name']
    # optional
    # surface 3d
    surface3d = dict_plot['surface3d'] if 'surface3d' in dict_plot.keys() else False
    xbound = dict_plot['xbound'] if 'xbound' in dict_plot.keys() else None
    ybound = dict_plot['ybound'] if 'ybound' in dict_plot.keys() else None
    method = dict_plot['method'] if 'method' in dict_plot.keys() else 'mesh'
    interp = dict_plot['interp'] if 'interp' in dict_plot.keys() else 'cubic'
    cbar_limit = dict_plot['cbar_limit'] if 'cbar_limit' in dict_plot.keys() else None
    cbar_label = dict_plot['cbar_label'] if 'cbar_label' in dict_plot.keys() else ''
    cmap = dict_plot['cmap'] if 'cmap' in dict_plot.keys() else cm.jet
    angle = dict_plot['angle'] if 'angle' in dict_plot.keys() else None
    lw_surf = dict_plot['lw_surf'] if 'lw_surf' in dict_plot.keys() else 0.2
    alpha_surf = dict_plot['alpha_surf'] if 'alpha_surf' in dict_plot.keys() else 0.5
    # contour 2d
    contour2d = dict_plot['contour2d'] if 'contour2d' in dict_plot.keys() else False
    fill = dict_plot['fill'] if 'fill' in dict_plot.keys() else False
    offset_per = dict_plot['offset_per'] if 'offset_per' in dict_plot.keys() else 0.0
    alpha_contour = dict_plot['alpha_contour'] if 'alpha_contour' in dict_plot.keys() else 0.5
    # scatter 3d
    scatter3d = dict_plot['scatter3d'] if 'scatter3d' in dict_plot.keys() else True
    size_scatter = dict_plot['size_scatter'] if 'size_scatter' in dict_plot.keys() else 5
    marker_scatter = dict_plot['marker_scatter'] if 'marker_scatter' in dict_plot.keys() else 'o'
    color_scatter = dict_plot['color_scatter'] if 'color_scatter' in dict_plot.keys() else 'k'
    alpha_scatter = dict_plot['alpha_scatter'] if 'alpha_scatter' in dict_plot.keys() else 1.0
    # scatter line 3d
    scatter_line = dict_plot['scatter_line'] if 'scatter_line' in dict_plot.keys() else False
    zbase = dict_plot['zbase'] if 'zbase' in dict_plot.keys() else 0
    ls_line = dict_plot['ls_line'] if 'ls_line' in dict_plot.keys() else '--'
    lw_line = dict_plot['lw_line'] if 'lw_line' in dict_plot.keys() else 1.0
    marker_line = dict_plot['marker_line'] if 'marker_line' in dict_plot.keys() else 'o'
    ms_line = dict_plot['ms_line'] if 'ms_line' in dict_plot.keys() else 0
    alpha_line = dict_plot['alpha_line'] if 'alpha_line' in dict_plot.keys() else 1.0
    # scatter 2d
    scatter2d = dict_plot['scatter2d'] if 'scatter2d' in dict_plot.keys() else False
    s_sct2d = dict_plot['s_sct2d'] if 's_sct2d' in dict_plot.keys() else 5
    marker_sct2d = dict_plot['marker_sct2d'] if 'marker_sct2d' in dict_plot.keys() else 'x'
    color_sct2d = dict_plot['color_sct2d'] if 'color_sct2d' in dict_plot.keys() else 'k'
    alpha_sct2d = dict_plot['alpha_sct2d'] if 'alpha_sct2d' in dict_plot.keys() else 1.0
    # plot
    if not (surface3d or scatter3d):
        raise ValueError('Either scatter 3D and/or contour 3D plot must be activated.')
    ax = dict_plot['ax']
    if (surface3d):
        if (method == 'mesh'):
            if (xbound):
                x1 = np.linspace(xbound[0], xbound[1], len(data))
            else:
                x1 = np.linspace(data[x_name].min(), data[x_name].max(), len(data))
            if (ybound):
                y1 = np.linspace(ybound[0], ybound[1], len(data))
            else:
                y1 = np.linspace(data[y_name].min(), data[y_name].max(), len(data))
            x2, y2 = np.meshgrid(x1, y1)
            z2 = scipy.interpolate.griddata((data[x_name], data[y_name]), data[z_name], (x2, y2), method=interp)
            if (cbar_limit):
                vmin=cbar_limit[0]
                vmax=cbar_limit[1]
            else:
                vmin=np.nanmin(z2)
                vmax=np.nanmax(z2)
            surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cmap, linewidth=lw_surf, antialiased=False, 
                                   alpha=alpha_surf, vmin=vmin, vmax=vmax)
        elif (method == 'tri'):
            surf = ax.plot_trisurf(data[y_name], data[x_name], data[z_name], cmap=cmap, linewidth=lw_surf)
        else:
            raise ValueError(f'{method} is not defined.')
        cb = plt.colorbar(surf, shrink=0.5, aspect=15)
        cb.set_label(cbar_label)
    if (contour2d):
        offset_base = np.nanmin(z2) - ((np.nanmax(z2) - np.nanmin(z2)) * offset_per)
        if (fill):
            ax.contourf(x2, y2, z2, zdir='z', offset=offset_base, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_contour)
        else:
            ax.contour(x2, y2, z2, zdir='z', offset=offset_base, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_contour)
    if (scatter3d):
        ax.scatter(data[x_name], data[y_name], data[z_name], s=size_scatter, marker=marker_scatter, color=color_scatter, alpha=alpha_scatter)
        if (scatter_line):
            for xi, yi, zi in zip(data[x_name], data[y_name], data[z_name]):        
                line=art3d.Line3D(*zip((xi, yi, zbase), (xi, yi, zi)), ls=ls_line, lw=lw_line, marker=marker_line, ms=ms_line, 
                                  markevery=(1,1), alpha=alpha_line)
                ax.add_line(line)
    if (scatter2d):
        ax.scatter(data[x_name], data[y_name], zs=zbase, zdir='z', s=s_sct2d, marker=marker_sct2d, color=color_sct2d, alpha=alpha_sct2d)
    if (angle):
        ax.view_init(angle[0], angle[1])
# END DEF
    
    

#%%
##################################
'''
# Matrix Plot Function
'''
##################################

#%% scatter matrix plot and/or contour 2d matrix plot
def scatterplot_matrix(**dict_plot):
    # parameters
    data = dict_plot['df_data']
    list_input = dict_plot['list_input']
    # optional
    z_name = dict_plot['z_name'] if 'z_name' in dict_plot.keys() else ''
    # fixed
    list_triangle = dict_plot['list_triangle']
    # contour
    contour = dict_plot['contour'] if 'contour' in dict_plot.keys() else False
    fill = dict_plot['fill'] if 'fill' in dict_plot.keys() else False
    xbound = dict_plot['xbound'] if 'xbound' in dict_plot.keys() else None
    ybound = dict_plot['ybound'] if 'ybound' in dict_plot.keys() else None
    interp = dict_plot['interp'] if 'interp' in dict_plot.keys() else 'cubic'
    cmap = dict_plot['cmap'] if 'cmap' in dict_plot.keys() else cm.jet
    # scatter
    scatter = dict_plot['scatter'] if 'scatter' in dict_plot.keys() else True
    size_scatter = dict_plot['size_scatter'] if 'size_scatter' in dict_plot.keys() else 5
    df_bound = dict_plot['df_bound'] if 'df_bound' in dict_plot.keys() else None
    # plot
    axes = dict_plot['axes']
    triangle = dict_plot['triangle']
    if (contour):
        df_z = data[[z_name]]
        vmax = df_z.max()
        vmin = df_z.min()
    for x, y in list_triangle:                
        x_name = list_input[x]
        y_name = list_input[y]
        if (contour):
            df_xy = data[[x_name, y_name]].to_numpy()
            list_z = []
            for i in range(len(df_z)):
                list_z.append(df_z[z_name].iloc[i])
            array_z = np.array(list_z)
            if (xbound):
                xi = np.linspace(xbound[0], xbound[1], len(data))
            else:
                xi = np.linspace(data[x_name].min(), data[x_name].max(), len(data))
            if (ybound):
                yi = np.linspace(ybound[0], ybound[1], len(data))
            else:
                yi = np.linspace(data[y_name].min(), data[y_name].max(), len(data))
            zi = scipy.interpolate.griddata(df_xy, array_z, (xi[None,:], yi[:,None]), method=interp)
            if (fill):
                axes[x,y].contourf(xi, yi, zi, cmap=cmap, vmax=vmax, vmin=vmin)
            else:
                axes[x,y].contour(xi, yi, zi, cmap=cmap, vmax=vmax, vmin=vmin)
            if (triangle == 'upper'):
                axes[y,x].set_axis_off()
            elif (triangle == 'lower'):
                axes[x,y].set_axis_off()
        if (scatter):
            axes[x,y].scatter(data[x_name], data[y_name], s=size_scatter)
        if (df_bound is not None):
            xtemp = df_bound[df_bound['design variable']==x_name]
            plt.xlim(xtemp['lower'].iloc[-1], xtemp['upper'].iloc[-1])
            ytemp = df_bound[df_bound['design variable']==y_name]
            plt.ylim(ytemp['lower'].iloc[-1], ytemp['upper'].iloc[-1]) 
# END DEF                   
        
        

#%%
##################################
'''
# Class Visualizer
'''
##################################

#%%
class Visualizer:
    def __init__(self, 
                 font_size: int = 8,
                 font_name: str = 'Arial',
                 ):
        """
        The base class for visualization.
        ----------
        Parameters
        ----------
        font_size: int, optional
            Font size, by default 8.
        font_name: str, optional
            Font family, by default Arial.
        """
        self.font_size: int = font_size
        self.font_name: str = font_name
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        matplotlib.rcParams['font.size'] = font_size
        self.dict_plotfunc_single = {'lineplot': lineplot,
                                     'scatterplot': scatterplot,
                                     'dendrogramplot': dendrogramplot,
                                     'heatmapplot': heatmapplot,
                                     'barplot': barplot,
                                     'boxplot': boxplot,
                                     'catplot': catplot}
        self.dict_plotfunc_single3d = {'scatter3Dplot': scatter3Dplot}
        self.dict_plotfunc_matrix = {'scatterplot_matrix': scatterplot_matrix}
        self.dict_plotfunc_template = {'correlogramplot': correlogramplot,
                                       'violinplot': violinplot}
    # END DEF
    
    #%%
    # return all plot functions available
    def printPlotFunc(self):
        print(f'plotSingle: {list(self.dict_plotfunc_single.keys())}')
        print(f'plotMultiple: {list(self.dict_plotfunc_single.keys())}')
        print(f'plotMatrix: {list(self.dict_plotfunc_matrix.keys())}')
        print(f'plotSingle3D: {list(self.dict_plotfunc_single3d.keys())}')
        print(f'plotTemplate: {list(self.dict_plotfunc_template.keys())}')
    # END DEF
        
    #%%
    # single plot in one figure
    def plotSingle(self, plotfunc, title='', dir_out='', figname='', figformat='.png', figsize=(6,3), dpi=300, show=False,
                   xlog='linear', ylog='linear', xlim=(None,None), ylim=(None,None), legend=True, legend_pos=[],
                   xlabel='', ylabel='', mark_vline=[], mark_hline=[], **dict_plot):
        # check plot function
        if (plotfunc not in self.dict_plotfunc_single.keys()):
            raise ValueError(f'Plotting function {plotfunc} is not defined. Use only {list(self.dict_plotfunc_single.keys())}.')
        # initialize
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_axisbelow(True)
        if (plotfunc == 'dendrogramplot'): 
            ax.yaxis.grid(True, lw=.5)
        else:
            ax.grid(b=True, lw=.5)
        
        # plot function
        dict_plot['ax'] = ax
        dict_plot['font_size'] = self.font_size
        dict_plot['keep_xlabel'] = dict_plot['keep_xlabel'] if 'keep_xlabel' in dict_plot.keys() else False
        dict_plot['keep_ylabel'] = dict_plot['keep_ylabel'] if 'keep_ylabel' in dict_plot.keys() else False
        self.dict_plotfunc_single[plotfunc](**dict_plot)
        
        # mark vertical and/or horizontal lines
        if (mark_vline):
            for thres_line in mark_vline:
                plt.axvline(x=thres_line, color='k', linestyle='--')
        if (mark_hline):
            for thres_line in mark_hline:
                plt.axhline(y=thres_line, color='k', linestyle='--')
        # diagramm properties
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if not (plotfunc == 'dendrogramplot' or plotfunc == 'boxplot' or dict_plot['keep_xlabel']):
            ax.set_xscale(xlog)
            ax.set_xlim(xlim)
        if not (dict_plot['keep_ylabel']):
            ax.set_yscale(ylog)
            ax.set_ylim(ylim)
        if (legend):
            if (legend_pos):
                ax.legend(bbox_to_anchor=(legend_pos[0], legend_pos[1]))
            else:
                ax.legend(labelspacing=0.1, loc='best')
        else:
            ax.get_legend().remove()
        fig.tight_layout()
        if (figname and dir_out):
            figname = figname + figformat
            filepath = os.path.join(dir_out, figname)
            plt.savefig(filepath, bbox_inches='tight')
        if (show):
            plt.show()
        else:
            plt.close(fig)
    # END DEF
    
    #%%
    # multiple plots in one figure
    def plotMultiple(self, plotfunc, fig_col, title='', dir_out='', figname='', figformat='.png', figsize=(6,3), dpi=300, show=False,
                     xlog='linear', ylog='linear', xlim=(None,None), ylim=(None,None), legend=True, ax_color=[],
                     xlabel='', ylabel='', mark_vline=[], mark_hline=[], **dict_plot):
        # check plot function
        if (plotfunc not in self.dict_plotfunc_single.keys()):
            raise ValueError(f'Plotting function {plotfunc} is not defined. Use only {list(self.dict_plotfunc_single.keys())}.')
        # initialize
        list_group = list(dict_plot['df_data'][dict_plot['label_group']].unique())
        fig_row = math.ceil(len(list_group) / fig_col)
        fig, axs = plt.subplots(nrows=fig_row, ncols=fig_col, figsize=figsize, dpi=dpi, sharex=False)
        
        for i, ax in enumerate(axs.flat):
            if (i >= len(list_group)):
                ax.set_axis_off()
                continue
            ax.set_axisbelow(True)
            ax.grid(b=True,lw=.5)
            
            # plot function
            dict_plot_temp = copy.deepcopy(dict_plot)
            dict_plot_temp['ax'] = ax
            df_temp = copy.deepcopy(dict_plot['df_data'])
            df_temp = df_temp[df_temp[dict_plot['label_group']] == list_group[i]]
            df_temp.reset_index(drop=True, inplace=True)
            dict_plot_temp['df_data'] = df_temp
            if ('df_fill' in dict_plot.keys()):
                df_temp = copy.deepcopy(dict_plot['df_fill'])
                df_temp = df_temp[df_temp[dict_plot['label_group']] == list_group[i]]
                df_temp.reset_index(drop=True, inplace=True)
                dict_plot_temp['df_fill'] = df_temp
            self.dict_plotfunc_single[plotfunc](**dict_plot_temp)
            
            # mark vertical and/or horizontal lines
            if (mark_vline):
                for thres_line in mark_vline:
                    ax.axvline(x=thres_line, color='k', linestyle='--')
            if (mark_hline):
                for thres_line in mark_hline:
                    ax.axhline(y=thres_line, color='k', linestyle='--')
            if (plotfunc not in ['boxplot', 'catplot']):
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xscale(xlog)
                ax.set_yscale(ylog)
            ax.set(xlabel=xlabel)
            ax.set(ylabel=ylabel)
            ax.set_title(list_group[i])
            ax.tick_params(axis='x')
            ax.tick_params(axis='y')
            if (legend):
                ax.legend(labelspacing=0.1, loc='best')
            else:
                ax.get_legend().remove()
            if (ax_color):
                if (ax_color[i]):
                    ax.set_facecolor(ax_color[i])
        
        # diagramm properties
        fig.suptitle(title)
        fig.tight_layout()
        if (figname and dir_out):
            figname = figname + figformat
            filepath = os.path.join(dir_out, figname)
            plt.savefig(filepath, bbox_inches='tight')
        if (show):
            plt.show()
        else:
            plt.close(fig)
    # END DEF
    
    #%%
    # n x n matrix plot in one figure
    def plotMatrix(self, plotfunc, title='', dir_out='', figname='', figformat='.png', figsize=(6,3), dpi=300, show=False,
                   triangle='both', histogram=False, histtype='bar', **dict_plot):
        # check plot function
        if (plotfunc not in self.dict_plotfunc_matrix.keys()):
            raise ValueError(f'Plotting function {plotfunc} is not defined. Use only {list(self.dict_plotfunc_matrix.keys())}.')
        if (triangle not in ['both', 'upper', 'lower']):
            raise ValueError(f'Variable {triangle} is not defined.')
        # initialize
        list_input = dict_plot['list_input']
        df = dict_plot['df_data'][list_input]
        numdata, numvars = df.shape
        fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        
        for ax in axes.flat:
            # Hide all ticks and labels
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            # Set up ticks only on one side for the "edge" subplots.
            if ax.is_first_col():
                ax.yaxis.set_ticks_position('left')
                ax.yaxis.set_label_position("left")
            if ax.is_last_col():
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_label_position("right")
            if ax.is_first_row():
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position("top")
            if ax.is_last_row():
                ax.xaxis.set_ticks_position('bottom')
                ax.xaxis.set_label_position("bottom")
        
        # Plot the data.
        dict_plot['axes'] = axes
        dict_plot['triangle'] = triangle
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            if (triangle == 'upper'):
                list_triangle = [(i,j)]
            elif (triangle == 'lower'):
                list_triangle = [(j,i)]
            elif (triangle == 'both'):
                list_triangle = [(i,j), (j,i)]
            else:
                raise ValueError(f'Variable {triangle} is not defined.')
            dict_plot['list_triangle'] = list_triangle
            self.dict_plotfunc_matrix[plotfunc](**dict_plot)
        
        # Diagonal
        for i, label in enumerate(list_input):
            if (histogram):
                df_bound = dict_plot['df_bound'] if 'df_bound' in dict_plot.keys() else None
                if (df_bound is not None):
                    xtemp = df_bound[df_bound['design variable']==label]
                    axes[i,i].hist(df[label], histtype=histtype, range=(xtemp['lower'].iloc[-1], xtemp['upper'].iloc[-1]))
                else:
                    axes[i,i].hist(df[label], histtype=histtype)
            else:
                axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
                if (triangle=='upper'):
                    axes[i,i].set_axis_off()
        
        
        # Turn on the proper x or y axes ticks.
        for i in range(numvars):
            for j in range(numvars):
                if (i!=j):
                    if (i==0 or i==numvars-1):
                        axes[i,j].xaxis.set_visible(True)
                    if (j==0 or j==numvars-1):
                        axes[i,j].yaxis.set_visible(True)
        # Set labels on first column and last row
        for i in range(len(list_input)):
            plt.setp(axes[-1, i], xlabel=list_input[i])
            plt.setp(axes[0, i], xlabel=list_input[i])
            plt.setp(axes[i, 0], ylabel=list_input[i])
            plt.setp(axes[i, -1], ylabel=list_input[i])
        # diagramm properties
        fig.suptitle(title)
        if (figname and dir_out):
            figname = figname + figformat
            filepath = os.path.join(dir_out, figname)
            plt.savefig(filepath, bbox_inches='tight')
        if (show):
            plt.show()
        else:
            plt.close(fig)
    # END DEF  
    
    #%%
    # 3D single plot in one figure
    def plotSingle3D(self, plotfunc, title='', dir_out='', figname='', figformat='.png', figsize=(6,3), dpi=300, show=False,
                     xlim=(None,None), ylim=(None,None), zlim=(None,None), 
                     xlabel='', ylabel='', zlabel='', **dict_plot):
        # check plot function
        if (plotfunc not in self.dict_plotfunc_single3d.keys()):
            raise ValueError(f'Plotting function {plotfunc} is not defined. Use only {list(self.dict_plotfunc_single3d.keys())}.')
        # initialize
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d') 
        
        # plot function
        dict_plot['ax'] = ax
        self.dict_plotfunc_single3d[plotfunc](**dict_plot)
        
        # diagramm properties
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        fig.tight_layout()
        if (figname and dir_out):
            figname = figname + figformat
            filepath = os.path.join(dir_out, figname)
            plt.savefig(filepath, bbox_inches='tight')
        if (show):
            plt.show()
        else:
            plt.close(fig)
    # END DEF
    
    #%%
    # blank/dummy framework
    def plotTemplate(self, plotfunc, title='', dir_out='', figname='', figformat='.png', figsize=(6,3), dpi=300, show=False,
                     **dict_plot):
        # check plot function
        if (plotfunc not in self.dict_plotfunc_template.keys()):
            raise ValueError(f'Plotting function {plotfunc} is not defined. Use only {list(self.dict_plotfunc_template.keys())}.')
        # plot function
        dict_plot['title'] = title
        dict_plot['dir_out'] = dir_out
        dict_plot['figname'] = figname
        dict_plot['figformat'] = figformat
        dict_plot['figsize'] = figsize
        dict_plot['dpi'] = dpi
        dict_plot['show'] = show
        self.dict_plotfunc_template[plotfunc](**dict_plot)
    # END DEF
# END CLASS
            
    
    




        


#%%








#%%









