import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    num_nodes = '3'
    specific_run = num_nodes + '_10_20_100_0.001_1000.0_False'
    result_folder = 'results_dim_' + num_nodes
    result_file = 'prediction_results.pkl'
    file_path = '/home/andrew/vcn_adapted/all_results/' + result_folder + '/prot_1/' + specific_run + '/' + result_file

    plot_name = 'elbo_full_epoch_dim_' + num_nodes 
    plot_path = 'plots/' + plot_name + '.pdf'
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

        prediction = data['prediction_averaged']
        binarized_mean_pred = (prediction>0.9).float()
        print(prediction)
        print(binarized_mean_pred)

        # elbo_val = np.array(data['elbo_val'])
        # elbo_train = np.array(data['elbo_train'])

        # plt.plot(elbo_train, color = 'b', alpha = 0.5, linewidth = 1)
        # plt.plot(elbo_val, color = 'r', linewidth = 1.1)
        # plt.legend(['ELBO train', 'ELBO validation'])
        # plt.show()

        # plt.savefig(plot_path)
        # print('saved following file: %s' % plot_name)

