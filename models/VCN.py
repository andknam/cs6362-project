import os, sys
import os.path as osp
import numpy as np
import torch
import argparse
from datetime import datetime
import pickle as pkl
import shutil
import networkx as nx
import time

import utils
import matplotlib.pyplot as plt
from models import vcn, autoreg_base, factorised_base, bge_model
from data import erdos_renyi, distributions
import graphical_models
from sklearn import metrics

def train(model, bge_train, optimizer, baseline, batch_size, e, device):
    kl_graphs = 0.
    losses = 0.
    likelihoods = 0.

    model.train()
   
    optimizer.zero_grad()
    likelihood, kl_graph, log_probs = model(batch_size, bge_train, e)  #TODO: Check if additional entropy regularization is required
    score_val = ( - likelihood + kl_graph).detach()
    per_sample_elbo = log_probs*(score_val-baseline)
    baseline = 0.95 * baseline + 0.05 * score_val.mean() 
    loss = (per_sample_elbo).mean()
    loss.backward()
    optimizer.step()
    
    likelihoods = -likelihood.mean().item()
    kl_graphs = kl_graph.mean().item() 
    losses = ( -likelihood  + kl_graph).mean().item()
        
    return  losses, likelihoods,  kl_graphs, baseline
    


def evaluate(model, bge_test, batch_size, e, device):
    model.eval()

    with torch.no_grad():
        
        likelihood, kl_graph, _ = model(batch_size, bge_test, e) 
        elbo = (-likelihood + kl_graph).mean().item()
        likelihoods = -likelihood.mean().item()
        
    return elbo, likelihoods
        
def load_model(args):
    if not args.no_autoreg_base:
        graph_dist = autoreg_base.AutoregressiveBase(args.num_nodes, device = args.device, temp_rsample = 0.1).to(args.device)
    else:
        graph_dist = factorised_base.FactorisedBase(args.num_nodes, device = args.device, temp_rsample = 0.1).to(args.device)
    
    def _gibbs_update(curr, epoch):
        if epoch < args.epochs*0.05:
            return curr
        else:
            return args.gibbs_temp_init+ (args.gibbs_temp - args.gibbs_temp_init)*(10**(-2 * max(0, (args.epochs - 1.1*epoch)/args.epochs)))
    
    if args.anneal:
        gibbs_update = _gibbs_update
    else:
        gibbs_update = None

    model = vcn.VCN(num_nodes = args.num_nodes, graph_dist = graph_dist, sparsity_factor = args.sparsity_factor, gibbs_temp_init = args.gibbs_temp_init, gibbs_update = gibbs_update).to(args.device)
    print(model, flush = True)
    return model

def load_data(args):
    model = load_model(args)

    optimizer = torch.optim.Adam(model.parameters() , args.lr)
    
    bge_train, train_data = load_data(args)
    if args.num_nodes <=4:
        g_dist = distributions.GibbsDAGDistributionFull(args.num_nodes, args.gibbs_temp, args.sparsity_factor)
    else:
        g_dist = distributions.GibbsUniformDAGDistribution(args.num_nodes, args.gibbs_temp, args.sparsity_factor)
    
    best_elbo = 1e20
    likelihood = []
    kl_graph = []
    elbo_train = []
    val_elbo = []
    baseline = 0.
    best_likelihood = 1e20
    best_kl = 1e20
    
    time_epoch = []
    if not args.eval_only:    
        for e in range(1, args.epochs + 1):
            temp_time = time.time()
            el, li, kl_g, baseline = train(model, bge_train, optimizer, baseline, args.batch_size, e, args.device)
            time_epoch.append(time.time()- temp_time)
            likelihood.append(li), kl_graph.append(kl_g), elbo_train.append(el)
            elbo_epoch, likelihood_epoch = evaluate(model, bge_train, args.batch_size, e, args.device)
            val_elbo.append(elbo_epoch)

            if e % 100 == 0:
                kl_full, hellinger_full = 0., 0.
                if args.num_nodes<=4:
                    kl_full, hellinger_full = full_kl_and_hellinger(model, bge_train, g_dist, args.device)

                print('Epoch {}:  TRAIN - ELBO: {:.5f} likelihood: {:.5f} kl graph: {:.5f} VAL-ELBO: {:.5f} Temp Target {:.4f} Time {:.2f}'.\
                    format(e, el, li,kl_g, elbo_epoch, model.gibbs_temp, np.sum(time_epoch[e-100:e]), flush = True))

                torch.save({'model':model.state_dict(), 'best_elbo':best_elbo, 'saved_epoch': e, 'time': time_epoch,\
                      'likelihood': likelihood, 'kl_graph': kl_graph, 'elbo_train': elbo_train, 'val_elbo': val_elbo, 'baseline': baseline}, osp.join(args.save_path, 'last_saved_model.pth'))
                
        torch.save({'model':model.state_dict(), 'best_elbo':best_elbo, 'saved_epoch': args.epochs, 'time': time_epoch,\
                    'likelihood': likelihood, 'kl_graph': kl_graph, 'elbo_train': elbo_train, 'val_elbo': val_elbo, 'baseline': baseline}, osp.join(args.save_path, 'best_model.pth'))

    model.load_state_dict(torch.load(osp.join(args.save_path,'best_model.pth'))['model'])
    shd, prc, rec = exp_shd(model, train_data.adjacency_matrix)
    kl_full = 0.
    hellinger_full = 0.
    auroc_score = 0.
    if args.num_nodes<=4:
        kl_full, hellinger_full = full_kl_and_hellinger(model, bge_train, g_dist, args.device)
    else:
        auroc_score = auroc(model, train_data.adjacency_matrix)

    print('Exp SHD:', shd,  'Exp Precision:', prc, 'Exp Recall:', rec, 'Kl_full:', kl_full, 'hellinger_full:', hellinger_full,\
    'auroc:', auroc_score)

    with open(osp.join(args.save_path, 'results.pkl'), 'wb') as bb:
            pkl.dump({'likelihood':likelihood, 'kl_graph':kl_graph, 'elbo_train': elbo_train,\
            'elbo_val':val_elbo, 'kl_best_full': kl_full, 'hellinger_best_full': hellinger_full, 'time': time_epoch\
            , 'baseline': baseline, 'exp_shd': shd, 'exp_prc': prc, 'exp_rec': rec, 'auroc': auroc_score}, bb)

if __name__ == '__main__':

    # get data 
    dataset_name = 'processed_data.txt'
    data = pd.read_csv('./data/'+dataset_name, sep=',')

    #get the proteins (DAG nodes data)
    prot_data = data.loc[:,'raf':'jnk']