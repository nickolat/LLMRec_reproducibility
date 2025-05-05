# How Powerful are LLMs to Support Multimodal Recommendation? A Reproducibility Study of LLMRec

This repository contains reproducibility and benchmarking codes for the LLMRec paper: https://arxiv.org/pdf/2311.00423

-----------
<h2> Datasets </h2>

`Netflix` [Original Split]: For replicability, dataset is available in LLMRec GitHub repository: https://github.com/HKUDS/LLMRec.git.
For reproducibility, check: ... (Files contained in data/Netflix/Train_Test have been created following the authors' pipeline).  

`Netflix` [Our Split]: For our benchmarking we used data available at: ... (path: data/Netflix/Train_Val_Test) 

`Amazon-DigitalMusic`: Original dataset available at https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html, 
processed with Ducho (GitHub repository: https://github.com/sisinflab/Ducho.git): 
1. Downloading of the Original Dataset (Digital_Music) via Ducho/demos/demo_recsys/download_amazon.sh
2. Processing of the dataset via Ducho/demos/demo_recsys/prepare_dataset.py with name='Digital_Music' 
and the meta dataset including also 'title' (two checks on its value should be added: NaN values or values with a string length = 0 are not allowed)

The already processed dataset is available at: ... (path: data/Amazon)

<h2> Usage </h2>

<h4> RQ2: Benchmarking with GPT-4 Turbo </h4>

Update `key` in `utilities.py` with your OpenAI key value and `endpoint` in `gpt4_x.py`

After having downloaded the paper's code, add the `gpt4_x.py` files in the directory `LLM_augmentation_construct_prompt`

```
python ./gpt4_feedback.py
python ./gpt4_user.py
python ./gpt4_item.py
```
<h4> RQ5: Topological properties of the LLM-augmented user-item graph </h4>


The computation of the following characteristics: `['space_size', 'shape', 'density', 'gini_user',
                            'gini_item', 'average_degree_users', 'average_degree_items',
                            'average_clustering_coefficient_dot_users',
                            'average_clustering_coefficient_dot_items', 'degree_assortativity_users',
                            'degree_assortativity_items']` is based on the code from the GitHub repository at https://github.com/sisinflab/Topology-Graph-Collaborative-Filtering.git

```
python Topology/check_dataset.py
python Topology/generate_only_characteristics.py
```
<h2> Hyperparameter study </h2>

In the following, the complete gridsearch of the Baselines on the chosen hyperparameters is reported in terms of Recall@20, nDCG@20 and Precision@20.
In bold, the best hyperparameters.

<h4>Netflix</h4>
The results in the table are reported after evaluation on the original test set provided by the authors' paper.

| Baseline | R@20   | N@20   | P@20   | Hyperparameter Search                                                                                                      |
|----------|--------|--------|--------|----------------------------------------------------------------------------------------------------------------------------|
| MF-BPR   | 0.0623 | 0.0246 | 0.0031 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**]                                                        |
| NGCF     | 0.0645 | 0.0265 | 0.0032 | lr: [0.0005, 0.001, 0.005, 0.01, **0.02**, 0.03, 0.04], l_w: [1e-5, 8e-3, **1e-2**]                                        |
| LightGCN | 0.0645 | 0.0239 | 0.0032 | lr: [0.0005, 0.001, 0.005, 0.01, **0.02**, 0.03, 0.04], l_w: [1e-5, **8e-3**, 1e-2]                                        |
| VBPR     | 0.0612 | 0.0205 | 0.0031 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**]                                                        |
| BM3      | 0.0726 | 0.0251 | 0.0036 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], reg_weight: [0.1, **0.01**]                                                  |
| MGCN     | 0.0645 | 0.0226 | 0.0032 | lr: [**0.0001**, 0.001, 0.01], c_l: [**0.001**, 0.01, 0.1]                                                                 |
| FREEDOM  | 0.0612 | 0.0214 | 0.0031 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [**1e-5**, 1e-2]                                                        |
| LATTICE  | 0.0656 | 0.0267 | 0.0033 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], regs: [1e-5, **1e-2**], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MICRO    | 0.0602 | 0.0194 | 0.0030 | lr: [0.0001, **0.0005**, 0.001, 0.005, 0.01], regs: [**1e-5**, 1e-2], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7     |
| MMSSL    | 0.0667 | 0.0212 | 0.0033 | lr generator: [**0.00055***, 4.5e-4, 5e-4, 5.4e-3, 5.6e-3], lr discriminator: [2.5e-4, **3e-4**, 3.5e-4]                           |
| SGL      | 0.0732 | 0.0273 | 0.0037 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [1e-5, **1e-2**]                                                                |

<h4>Amazon-Music</h4>

| Baseline | R@20   | N@20   | P@20   | Hyperparameter Search |
|----------|--------|--------|--------|---------------|
| MF-BPR   | 0.2868 | 0.1610 | 0.0303 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [1e-5, **1e-2**] |
| NGCF     | 0.2756 | 0.1528 | 0.0292 | lr: [0.0005, 0.001, 0.005, 0.01, 0.02, **0.03**, 0.04], l_w: [1e-5, 8e-3, **1e-2**] |
| LightGCN | 0.3009 | 0.1664 | 0.0318 | lr: [0.0001, 0.0005, 0.001, 0.005, 0.01, **0.02**, 0.03, 0.04], l_w: [1e-5, **8e-3**, 1e-2] |
| VBPR     | 0.2801 | 0.1590 | 0.0294 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**] |
| BM3      | 0.2958 | 0.1621 | 0.0309 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], reg_weight: [**0.1**, 0.01] |
| MGCN     | 0.2764 | 0.1602 | 0.0296 | lr: [**0.0001**, 0.001, 0.01], c_l: [0.001, 0.01, **0.1**] |
| FREEDOM  | 0.3131 | 0.1769 | 0.0332 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [**1e-5**, 1e-2] |
| LATTICE  | 0.3215 | 0.2202 | 0.0338 | lr: [0.0001, 0.0005, **0.001**, 0.005, 0.01], regs: [1e-5, **1e-2**], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MICRO    | 0.3308 | 0.2272 | 0.0345 | lr: [0.0001, 0.0005, **0.001**, 0.005, 0.01], regs: [1e-5, **1e-2**], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MMSSL    | 0.3211 | 0.2168 | 0.0340 | lr generator: [0.00055*, **4.5e-4**, 5e-4, 5.4e-3, 5.6e-3], lr discriminator: [**2.5e-4**, 3e-4, 3.5e-4] |
| SGL      | 0.2969 | 0.1733 | 0.0321 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**] |

*: 0.00055 has been paired only with 3e-4




=======
# LLMRec_reproducibility
>>>>>>> upstream/main
