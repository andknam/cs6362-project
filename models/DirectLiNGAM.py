#Marco Barbero Mota
#November 2022
#This script makes significant use of the
#DirectLiNGAM package by Shohei Shimizu et al.

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pandas as pd
import graphviz
import os
import lingam
from lingam.utils import make_dot
from utils import *
import networkx as nx

# %%
def get_pvalue_heatmap(model, data, plot_heatmap = False, heatmap_filename = None):
#p-values of independence between nodes
#This step checks whether LiNGAM assumption of independence is correct
    p_values = model.get_error_independence_p_values(data)

    if plot_heatmap == True:
        matrix2heatmap(p_values,x_labels=data.columns.values, y_labels=data.columns.values,
        color_bar_legend='p-value', heatmap_filename=heatmap_filename,color_bar=True)
        return p_values

def to_0_1(matrix):
    return (matrix+1)/2

def matrix2heatmap(matrix, x_labels = None, y_labels = None,color_bar_legend = ' ', 
    heatmap_filename = None,color_bar = False, discrete_heatmap = False, cmap='hot'):
    
    fig,ax = plt.subplots(figsize = (6,6))

    if discrete_heatmap == True:
        # make a color map of fixed colors
        cmap = colors.ListedColormap(['gray', 'forestgreen', 'red'])

        T = ax.imshow(to_0_1(matrix), cmap=cmap, interpolation=None, vmin = 0, vmax=1)

        legend_handles = [Patch(facecolor='gray', edgecolor='k', label='Missing edge'),
            Patch(facecolor='forestgreen',edgecolor='k' ,  label='Correct edge'),
            Patch(facecolor='red',edgecolor='k',  label='Extra edge')]
        
        plt.legend(handles=legend_handles, ncol=3, bbox_to_anchor=[0.5, 1.02], 
            loc='lower center', fontsize=12.5, handlelength=.8, frameon = True)
    else:
        T = ax.imshow(matrix, cmap=cmap, interpolation=None)

    if color_bar == True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(T,cax=cax)
        cbar.set_label(color_bar_legend,labelpad = 25,fontsize = 14, rotation = 270)

    if all((x_labels!=None) & (y_labels!=None)):
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    if heatmap_filename!=None:
        plt.savefig(heatmap_filename, dpi = 150)

def single_LiNGAM_model(data, DAG_file_name='Single model DAG'):
#Single DirectLiNGAM model -> example to illustrate
    model = lingam.DirectLiNGAM()
    model.fit(data)

    #plot and save the DAG
    labels = [str(prot) for prot in data.columns]
    single_model_DAG = make_dot(model.adjacency_matrix_, labels=labels)
    single_model_DAG.render(DAG_file_name)

    return model

def bootstrapping(data, n_bootstrap = 500, save_graph = False, file_name = None):
# Perform 500 (default: same number as Sachs et al.) bootstrap models and average them
# obtain edge probabilities
    lingam_model = lingam.DirectLiNGAM()
    bootstrapped_model = lingam_model.bootstrap(data, n_sampling = n_bootstrap)

    if save_graph==True:
        labels = [str(prot) for prot in data.columns]
        boot_DAG = make_dot(bootstrapped_model.get_probabilities(), labels = labels)
        boot_DAG.render(file_name)

    return bootstrapped_model, lingam_model

def bootstrapped_posterior_over_DAGs(boot_model):
    '''
    Returns both the overall dfictionary 
    as well as the MAP adjacency matrix
    '''

    posterior_dict = bootstrapped_model.get_directed_acyclic_graph_counts()
    total_DAGs = sum(posterior_dict['count'])
    total_diff_DAGs = len(posterior_dict['count'])
    posterior_dict['Posterior probability'] = [posterior_dict['count'][i]/total_DAGs for i in range(total_diff_DAGs)]

    return posterior_dict, get_adjacency_from_DAG_dict(posterior_dict['dag'][0])

def get_adjacency_from_DAG_dict(dict_post):
    d1 = dict_post['from']
    d2 = dict_post['to']
    tuple_list = list(zip(d1,d2)) 
    G = nx.Graph()
    G.add_edges_from(tuple_list)
    return nx.to_numpy_array(G, dtype=float)

def DAG_accuracy(matrix,GT):
    n_entries = matrix.shape[0] * matrix.shape[1]
    accuracy = 1-(sum(sum(abs(matrix-GT)))/n_entries)
    return accuracy

#%%
if __name__ == '__main__':

    #experiment data 
    dataset_name = 'processed_data.txt'
    #dataset_name = 'orig_data.txt'

    #get data
    data = pd.read_csv('../protein_data/'+dataset_name, sep=',')

    #get the proteins (DAG nodes data)
    prot_data = data.loc[:,'raf':'jnk']

    #get the treatments (changes to the DAG) data -> we standardize based on these
    #to remove their effect, so we dont need this data for the overall 
    # DAG discovery in this method, although it could also be leveraged
    treatment_data = data.iloc[:,-9:]

    # %%
    bootstrapped_model, lingam_model = bootstrapping(data = prot_data,
    n_bootstrap=500,
    save_graph=True,
    file_name='500_bootstrap')
    #%%
    p_values = get_pvalue_heatmap(lingam_model,prot_data,plot_heatmap=True, 
        heatmap_filename='p-value heatmap.pdf')
    #%%
    #Mean bootstrapping adjancency matrix
    mean_adjacency = np.mean(bootstrapped_model.adjacency_matrices_, axis = 0)
    #0.85 threshold (Sachs et al.) mean matrix
    binarized_mean_pred = (mean_adjacency>0.85).astype(float)

    #posterior dictionary and MAP predicted adjacency matrix
    posterior_dict, MAP_pred = bootstrapped_posterior_over_DAGs(bootstrapped_model)

    #Get ground truths
    sup_GT = get_supplemented_ground_truth_adj_matrix()[0]
    # %%
    #Plot differences with the GT
    #MAP
    matrix2heatmap(MAP_pred-sup_GT,x_labels=prot_data.columns.values, 
        y_labels=prot_data.columns.values,
        color_bar_legend=None, heatmap_filename='MAP-GT.pdf',
        color_bar=False, discrete_heatmap = True, cmap = 'Blues')
    print('MAP accuracy: ', DAG_accuracy(MAP_pred,sup_GT))
    #Binarized mean
    matrix2heatmap(binarized_mean_pred-sup_GT,x_labels=prot_data.columns.values, 
        y_labels=prot_data.columns.values,
        color_bar_legend=None, heatmap_filename='85_mean-GT.pdf',
        color_bar=False, discrete_heatmap = True, cmap = 'Blues')
    print('Binarized mean accuracy: ', DAG_accuracy(binarized_mean_pred,sup_GT))
    # %%
