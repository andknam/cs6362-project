Adapted version of Pytorch implementation of Variational Causal Networks for use with Sachs et al. protein dataset

## Installation
You can install the dependencies using 
`pip install -r requirements.txt
`

## Examples

`python main.py --num_nodes [num_nodes] --epochs [num_epochs] --save_path [save_path]`

`python main.py --num_nodes 11 --epochs 1000 --data_type prot -save_path ../../vcn_results/all_results/ --early_stop`

## Citation

[Yashas Annadani](https://yashasannadani.com), [Jonas Rothfuss](https://las.inf.ethz.ch/people/jonas-rothfuss), [Alexandre Lacoste](https://ca.linkedin.com/in/alexandre-lacoste-4032465), [Nino Scherrer](https://ch.linkedin.com/in/ninoscherrer), [Anirudh Goyal](https://anirudh9119.github.io/), [Yoshua Bengio](https://mila.quebec/en/yoshua-bengio/), [Stefan Bauer](https://www.is.mpg.de/~sbauer)

	@article{annadani2021variational,
	title={Variational Causal Networks: Approximate Bayesian Inference over Causal Structures},
	author={Annadani, Yashas and Rothfuss, Jonas and Lacoste, Alexandre and Scherrer, Nino and Goyal, Anirudh and Bengio, Yoshua and Bauer, Stefan},
	journal={arXiv preprint arXiv:2106.07635},
	year={2021}
	}
	
 
