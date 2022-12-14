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
from data import erdos_renyi, distributions, sachs_protein
import graphical_models
from sklearn import metrics

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Variational Causal Networks')
    parser.add_argument('--save_path', type=str, default = 'results_anneal/',
                    help='Path to save result files')
    parser.add_argument('--epochs', type=int, default=30000,
                    help='Number of iterations to train')
    parser.add_argument('--num_nodes', type=int, default=2,
                    help='Number of nodes in the causal model')
    parser.add_argument('--data_type', type=str, default='er',
                    help='Type of data')
    parser.add_argument('--early_stop', action='store_true', default=False,
                    help='Perform early stopping')
    # parser.add_argument('--no_autoreg_base', action='store_true', default=False,
    #                 help='Use factorisable disrtibution')
    # parser.add_argument('--seed', type=int, default=10,
    #                 help='random seed (default: 10)')
    # parser.add_argument('--data_seed', type=int, default=20,
    #                 help='random seed for generating data(default: 20)')
    # parser.add_argument('--batch_size', type=int, default=1000,
    #                 help='Batch Size for training')
    # parser.add_argument('--lr', type=float, default=1e-2,
    #                 help='Learning rate')
    # parser.add_argument('--gibbs_temp', type=float, default=1000.0,
    #                 help='Temperature for the Graph Gibbs Distribution')
    # parser.add_argument('--sparsity_factor', type=float, default=0.001,
    #                 help='Hyperparameter for sparsity regularizer')
    # parser.add_argument('--num_samples', type=int, default=100,
    #                 help='Total number of samples in the synthetic data')
    # parser.add_argument('--noise_type', type=str, default='isotropic-gaussian',
    #                 help='Type of noise of causal model')
    # parser.add_argument('--noise_sigma', type=float, default=1.0,
    #                 help='Std of Noise Variables')
    # parser.add_argument('--theta_mu', type=float, default=2.0,
    #                 help='Mean of Parameter Variables')
    # parser.add_argument('--theta_sigma', type=float, default=1.0,
    #                 help='Std of Parameter Variables')
    # parser.add_argument('--exp_edges', type=float, default=1.0,
    #                 help='Expected number of edges in the random graph')
    # parser.add_argument('--eval_only', action='store_true', default=False,
    #                 help='Perform Just Evaluation')
    # parser.add_argument('--anneal', action='store_true', default=False,
    #                 help='Perform gibbs temp annealing')

    args = parser.parse_args()
    args.data_size = args.num_nodes * (args.num_nodes-1)
    root = args.save_path
    list_dir = os.listdir(args.save_path)
    args.save_path = os.path.join(args.save_path, args.data_type + '_' + str(int(args.exp_edges)), str(args.num_nodes) + '_' + str(args.seed) + '_' + str(args.data_seed) + '_' + str(args.num_samples) + '_' + \
      str(args.sparsity_factor) +'_' + str(args.gibbs_temp) + '_' + str(args.no_autoreg_base)) 
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.num_nodes == 2:
        args.exp_edges = 0.8

    args.gibbs_temp_init = 10.
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args

def sample_model_posterior(model,num_samples=1000):
    bs = 10000
    i = 0
    samples = []
    with torch.no_grad():
        while i<num_samples:
            curr = min(bs, num_samples-i)
            samples.append(model.graph_dist.sample([curr]).cpu().numpy().squeeze())
            i+=curr
    return samples

def exp_shd(model, ground_truth, num_samples = 1000):
    """Compute the Expected Structural Hamming Distance of the model"""
    shd = 0
    prc = 0.
    rec = 0.
    with torch.no_grad():
        samples = model.graph_dist.sample([num_samples])
        G = utils.vec_to_adj_mat(samples, model.num_nodes) 
        for i in range(num_samples):
            metrics = utils.shd(G[i].cpu().numpy(), ground_truth)
            shd += metrics['shd']
            prc += metrics['prc']
            rec += metrics['rec']
    return shd/num_samples, prc/num_samples, rec/num_samples
 
def full_kl_and_hellinger(model, bge_train, g_dist, device):
    """Compute the KL Divergence and Hellinger distance in lower dimensional settings (d<=4)"""

    bs = 100000
    all_adj = utils.all_combinations(model.num_nodes, return_adj = True).astype(np.float32)
    all_adj_vec = utils.all_combinations(model.num_nodes, return_adj = False).astype(np.float32)
    log_posterior_graph = torch.zeros(len(all_adj))
    log_prob_g = torch.zeros(len(all_adj))
    log_prob_model = torch.zeros(len(all_adj))
    with torch.no_grad():
        for tt in range(0,len(all_adj),bs):
            log_posterior_graph[tt:tt+bs] = bge_train.log_marginal_likelihood_given_g(w = torch.tensor(all_adj[tt:tt+bs]).to(device)).cpu() #Unnormalized Log Probabilities
            log_prob_model[tt:tt+bs] = model.graph_dist.log_prob(torch.tensor(all_adj_vec[tt:tt+bs]).to(device).unsqueeze(2)).cpu().squeeze()
        for tt in range(len(all_adj)):
            log_prob_g[tt] = g_dist.unnormalized_log_prob(g=all_adj[tt])
    graph_p = torch.distributions.categorical.Categorical(logits = log_posterior_graph + log_prob_g)
    graph_q = torch.distributions.categorical.Categorical(logits = log_prob_model)
    hellinger = (1./np.sqrt(2)) * torch.sqrt((torch.sqrt(graph_p.probs) - torch.sqrt(graph_q.probs)).pow(2).sum()) 
    return torch.distributions.kl.kl_divergence(graph_q, graph_p).item(), hellinger.item()

def train(model, bge_train, optimizer, baseline, batch_size, e, device, all_epoch_gradients):
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
    if args.data_type == 'd4':
        train_data = data_map[args.data_type](file_name = int(args.exp_edges))

    elif args.data_type == 'prot':
        # choose proteins restricted to number of nodes 
        protein_index_hashmap = {
            0:'raf',
            1:'mek',
            2:'plc',
            3:'pip2',
            4:'pip3',
            5:'erk',
            6:'akt',
            7:'pka',
            8:'pkc',
            9:'p38',
            10:'jnk'
        }
        
        protein_indices = [0, 3, 4, 9, 6, 10, 2, 5, 1, 7, 8] # randomly generated index list
        protein_names = [protein_index_hashmap[protein_indices[i]] for i in range(args.num_nodes)]
        protein_mapping = dict(zip(protein_names, range(args.num_nodes)))

        connections = {
            'raf': ['mek','akt'],
            'mek': ['erk','jnk','plc'],
            'plc': ['pip2','mek','pkc', 'pka', 'akt'],
            'pip2': ['pkc'],
            'pip3': ['pip2', 'akt','plc'],
            'erk': ['akt'],
            'akt': ['mek','pkc','raf','plc'],
            'pka': ['raf', 'akt', 'erk', 'jnk', 'mek', 'p38','plc','pkc'],
            'pkc': ['raf', 'jnk', 'mek', 'p38','akt','pka'],
            'p38': ['jnk'],
            'jnk': ['p38','mek']
        }

        train_data = data_map[args.data_type](num_nodes = args.num_nodes, connections = connections, mapping = protein_mapping, protein_names = protein_names)
        print(train_data.adjacency_matrix.shape)

    else:
        train_data = data_map[args.data_type](num_nodes = args.num_nodes, exp_edges = args.exp_edges, noise_type = args.noise_type, noise_sigma = args.noise_sigma, \
            num_samples = args.num_samples, mu_prior = args.theta_mu, sigma_prior = args.theta_sigma, seed = args.data_seed)

    if args.data_type == 'prot':
        # sachs protein data  
        dataset_name = 'processed_data.txt'
        data = pd.read_csv('../protein_data/'+dataset_name, sep=',')
        prot_data = torch.tensor(data.loc[:,protein_names].values)
        print(prot_data.shape)

        bge_train = bge_model.BGe(mean_obs = [args.theta_mu]*args.num_nodes, alpha_mu = 1.0, alpha_lambd=args.alpha_lambd, data = prot_data, device = args.device)

    else:
        bge_train = bge_model.BGe(mean_obs = [args.theta_mu]*args.num_nodes, alpha_mu = 1.0, alpha_lambd=args.alpha_lambd, data = train_data.samples, device = args.device)

    return bge_train, train_data

# perform early stopping if average change of previous 300 epochs is below threshold
def early_stop(elbo_train, previous_change):
    percent_change = abs(float((elbo_train[-1] - elbo_train[-2]) / elbo_train[-2]))

    if len(previous_change) > 300:
        previous_average = sum(previous_change[-300:]) / 300
        print('average percent change: %.7f' % previous_average, end = '\n')

        if previous_average <= 0.00045:
            print('BREAK, epoch: %.5f' % e)
            return True

    previous_change.append(percent_change)

def main(args):
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
    all_epoch_gradients = []
    previous_change = []
    
    time_epoch = []
    if not args.eval_only:    
        for e in range(1, args.epochs + 1):
            # break if early stop condition is satisfied
            if args.early_stop and len(elbo_train) > 2 and early_stop(elbo_train, previous_change):
                break

            temp_time = time.time()
            el, li, kl_g, baseline = train(model, bge_train, optimizer, baseline, args.batch_size, e, args.device, all_epoch_gradients)
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

        with open(osp.join(args.save_path, 'elbo_results.pkl'), 'wb') as bb:
            pkl.dump({'likelihood':likelihood, 'kl_graph':kl_graph, 'elbo_train': elbo_train,\
            'elbo_val':val_elbo, 'time': time_epoch, 'baseline': baseline}, bb)

        torch.save({'model':model.state_dict(), 'best_elbo':best_elbo, 'saved_epoch': args.epochs, 'time': time_epoch,\
                    'likelihood': likelihood, 'kl_graph': kl_graph, 'elbo_train': elbo_train, 'val_elbo': val_elbo, 'baseline': baseline}, osp.join(args.save_path, 'best_model.pth'))

    model.load_state_dict(torch.load(osp.join(args.save_path,'best_model.pth'))['model'])
    shd, prc, rec = exp_shd(model, train_data.adjacency_matrix)
    kl_full = 0.
    hellinger_full = 0.

    num_samples = 20000
    samples = model.graph_dist.sample([num_samples])
    prediction = utils.vec_to_adj_mat(samples, model.num_nodes)

    with open(osp.join(args.save_path, 'prediction_results.pkl'), 'wb') as bb:
        pkl.dump({'full_prediction': prediction, 'prediction_averaged': prediction.mean(axis=0)}, bb)

    print('Exp SHD:', shd,  'Exp Precision:', prc, 'Exp Recall:', rec, 'Kl_full:', kl_full, 'hellinger_full:', hellinger_full,\
    'auroc:', auroc_score)

    with open(osp.join(args.save_path, 'results.pkl'), 'wb') as bb:
            pkl.dump({'likelihood':likelihood, 'kl_graph':kl_graph, 'elbo_train': elbo_train,\
            'elbo_val':val_elbo, 'kl_best_full': kl_full, 'hellinger_best_full': hellinger_full, 'time': time_epoch\
            , 'baseline': baseline, 'exp_shd': shd, 'exp_prc': prc, 'exp_rec': rec, 'auroc': auroc_score}, bb)

if __name__ == '__main__':
    args = parse_args()
    if args.num_nodes <=4:
        args.alpha_lambd = 10.
    else:
        args.alpha_lambd = 1000.
    data_map = {'er': erdos_renyi.ER, 'prot': sachs_protein.PT}
    main(args)
