# Adapted Version of Variational Causal Networks
 Adapted version of Pytorch implementation of [Variational Causal Networks: Approximate Bayesian Inference over Causal Structures](https://arxiv.org/abs/2106.07635) (Annadani et al. 2021).
 
[Yashas Annadani](https://yashasannadani.com), [Jonas Rothfuss](https://las.inf.ethz.ch/people/jonas-rothfuss), [Alexandre Lacoste](https://ca.linkedin.com/in/alexandre-lacoste-4032465), [Nino Scherrer](https://ch.linkedin.com/in/ninoscherrer), [Anirudh Goyal](https://anirudh9119.github.io/), [Yoshua Bengio](https://mila.quebec/en/yoshua-bengio/), [Stefan Bauer](https://www.is.mpg.de/~sbauer)
 

## Installation
You can install the dependencies using 
`pip install -r requirements.txt
`

Create Directory structure which looks as follows: `[save_path]/er_1/`
If using the protein data set: `[save_path]/prot_1/`

## Examples

Run

`python main.py --num_nodes [num_nodes] --epochs 10000 --save_path [save_path]`

`python main.py --num_nodes [num_nodes] --epochs 1000 --data_type prot -save_path [save_path] --early_stop`

	@article{annadani2021variational,
	title={Variational Causal Networks: Approximate Bayesian Inference over Causal Structures},
	author={Annadani, Yashas and Rothfuss, Jonas and Lacoste, Alexandre and Scherrer, Nino and Goyal, Anirudh and Bengio, Yoshua and Bauer, Stefan},
	journal={arXiv preprint arXiv:2106.07635},
	year={2021}
	}
