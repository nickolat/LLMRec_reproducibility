# How Powerful are LLMs to Support Multimodal Recommendation? A Reproducibility Study of LLMRec

This repository contains reproducibility and benchmarking codes for the [LLMRec paper](https://doi.org/10.1145/3616855.3635853).

-----------
<h2> Datasets </h2>

Data storage (anonymous): https://drive.google.com/file/d/1ktu5GOBoL0uUrdM70EQVXHZpMc3LRctB/view?usp=sharing

`Netflix` [Original Split]: For replicability, dataset is available in LLMRec GitHub repository: https://github.com/HKUDS/LLMRec.git.
For reproducibility, check the Google Drive link above (Files contained in ./data/Netflix/Train_Test have been created following the authors' pipeline).  

`Netflix` [Our Split]: For our benchmarking we used data available at the Google Drive link above (path: ./data/Netflix/Train_Val_Test) 

`Amazon-DigitalMusic`: Original dataset available at https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html, processed with Ducho (GitHub repository: https://github.com/sisinflab/Ducho.git): 
1. Downloading of the Original Dataset (Digital_Music) via Ducho/demos/demo_recsys/download_amazon.sh
2. Processing of the dataset via Ducho/demos/demo_recsys/prepare_dataset.py with name='Digital_Music' and the meta dataset including also 'title' (two checks on its value should be added: NaN values or values with a string length = 0 are not allowed)

The already processed dataset is available at the Google Drive link above (path: ./data/Amazon).

To run each experiment, you have to put all the necessary input data in:
```
├─ LLMRec/
    ├── data/
        ├── netflix/
        ├── amazon/
```

<h2> Usage </h2>

<h4> RQ1: Replicability and reproducibility study </h4>

For LLMRec replicability: 
- We used the original code in the official repository, specifically the
latest commit available ([Jun 10, 2024](https://github.com/HKUDS/LLMRec/tree/169f361408dedc2334b6aac9ff7a8d016cc84230)).
- To use ‘netflix’ dataset name, change the name in line 71 of `main.py` original file.
- Run:   
  ```
   cd LLMRec/
   python ./main.py --dataset netflix
   ```

For LLMRec reproducibility from scratch:
1. First, use the original LATTICE or MMSSL implementations (available in LLMRec repository) to obtain the `candidate_indices`,
as indicated [here](https://github.com/HKUDS/LLMRec?tab=readme-ov-file#-candidate-preparing-for-llm-based-implicit-feedback-augmentation)
2. Add the `LLM_aug_unimodal` directory to LLMRec
3. In `utils.py`, set your keys and endpoints to use `gpt-35-turbo-16k` LLM and 
set `dataset = 'netflix'`, `llm = 'gpt35'`
4. Run:
    ```
    # LLM-based Data Augmentation
    cd LLMRec/LLM_aug_unimodal/
    python ./llm_feedback.py
    python ./llm_user.py
    python ./llm_item.py
    
    # Recommender training with LLM-augmented Data
    cd LLMRec/
    python ./main.py --dataset netflix
    ```

Note: we used Microsoft Azure AI platform to access all LLMs. 


<h4> RQ2: Benchmarking with LLama </h4>

1. Use codes in `LLM_aug_unimodal`
2. Set your keys and endpoints in `utils.py` to use `Meta-Llama-3.1-405B-Instruct` LLM
and set: `dataset = 'netflix'`, `llm = 'llama'`
3. Run:
    ```
    # LLM-based Data Augmentation
    cd LLMRec/LLM_aug_unimodal/
    python ./llm_feedback.py
    python ./llm_user.py
    python ./llm_item.py
   
    # Recommender training with LLM-augmented Data
    cd LLMRec/
    python ./main.py --dataset netflix
    ```

<h4> RQ2: Benchmarking with GPT-4 Turbo </h4>

1. Add the `LLM_aug_multimodal` directory to LLMRec
2. Update `key` and `endpoint` in `utilities.py` with your own subscription key and endpoint values
3. Run:
    ```
    # LLM-based Data Augmentation
    cd LLMRec/LLM_aug_multimodal/
    python ./gpt4_feedback.py
    python ./gpt4_user.py
    python ./gpt4_item.py
   
    # Recommender training with LLM-augmented Data
    cd LLMRec/
    python ./main.py --dataset netflix
    ```

<h4> RQ3: Benchmarking with new baselines and RLMRec </h4>

- To run the new recommendation baselines, use: https://github.com/sisinflab/Graph-Missing-Modalities
- To run RLMRec, use the official repository: https://github.com/HKUDS/RLMRec

<h4> RQ4: Benchmarking on Amazon-music dataset </h4>

In order to execute LLMRec with the Amazon-music dataset and `gpt-35-turbo-16k`:
1. Use codes in `LLM_aug_unimodal`
2. In `utils.py` set: `dataset = 'amazon'`, `llm = 'gpt35'`
3. Add the following lines of code after line 72 in the original `main.py` file:
    ```
    elif args.dataset=='amazon':
        augmented_total_embed_dict = {'title':[] , 'genres':[], 'artist':[], 'country':[], 'language':[]}
    ```
4. Run:
    ```
    # LLM-based Data Augmentation
    cd LLMRec/LLM_aug_unimodal/
    python ./llm_feedback.py
    python ./llm_user.py
    python ./llm_item.py
   
    # Recommender training with LLM-augmented Data
    cd LLMRec/
    python ./main.py --dataset amazon
    ```

<h4> RQ5: Topological properties of the LLM-augmented user-item graph </h4>

The code for the computation of the following characteristics: `['space_size', 'shape', 'density', 'gini_user',
                            'gini_item', 'average_degree_users', 'average_degree_items',
                            'average_clustering_coefficient_dot_users',
                            'average_clustering_coefficient_dot_items', 'degree_assortativity_users',
                            'degree_assortativity_items']` is contained in the Topology directory and based on the GitHub repository at https://github.com/sisinflab/Topology-Graph-Collaborative-Filtering.git

```
python Topology/check_dataset.py
python Topology/generate_only_characteristics.py
```
The already processed and analyzed data are available at the Google Drive link above (path: ./data_Topology)

<h2> Hyperparameter study </h2>

In the following, the complete gridsearch of the Baselines on the chosen hyperparameters is reported in terms of Recall@20, nDCG@20 and Precision@20.
In bold, the best hyperparameters.

Note: to use the validation set instead of test set in the original implementations,
without additional codes, you have to replace `self.test(users_to_test, is_val=False)` with
`self.test(users_to_val, is_val=True)` in the `main.py` file.


<h4>Netflix</h4>
The results in the table are reported after evaluation on the original test set provided by the authors' paper.

| Baseline  | R@20   | N@20   | P@20   | Hyperparameter Search                                                                                                          |
|-----------|--------|--------|--------|--------------------------------------------------------------------------------------------------------------------------------|
| MF-BPR    | 0.0623 | 0.0246 | 0.0031 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**]                                                            |
| NGCF      | 0.0645 | 0.0265 | 0.0032 | lr: [0.0005, 0.001, 0.005, 0.01, **0.02**, 0.03, 0.04], l_w: [1e-5, 8e-3, **1e-2**]                                            |
| LightGCN  | 0.0645 | 0.0239 | 0.0032 | lr: [0.0005, 0.001, 0.005, 0.01, **0.02**, 0.03, 0.04], l_w: [1e-5, **8e-3**, 1e-2]                                            |
| VBPR      | 0.0612 | 0.0205 | 0.0031 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**]                                                            |
| BM3       | 0.0726 | 0.0251 | 0.0036 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], reg_weight: [0.1, **0.01**]                                                      |
| MGCN      | 0.0645 | 0.0226 | 0.0032 | lr: [**0.0001**, 0.001, 0.01], c_l: [**0.001**, 0.01, 0.1]                                                                     |
| FREEDOM   | 0.0612 | 0.0214 | 0.0031 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [**1e-5**, 1e-2]                                                            |
| LATTICE*  | 0.0656 | 0.0267 | 0.0033 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], regs: [1e-5, **1e-2**], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MICRO*    | 0.0602 | 0.0194 | 0.0030 | lr: [0.0001, **0.0005**, 0.001, 0.005, 0.01], regs: [**1e-5**, 1e-2], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MMSSL*    | 0.0667 | 0.0212 | 0.0033 | lr generator: [**0.00055****, 4.5e-4, 5e-4, 5.4e-3, 5.6e-3], lr discriminator: [2.5e-4, **3e-4**, 3.5e-4]                      |
| SGL       | 0.0732 | 0.0273 | 0.0037 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [1e-5, **1e-2**]                                                            |
| *RLMRec** | 0.0683 | 0.0231 | 0.0034 | layer_num: [1,2,3,**4**], reg_weight: [0.5e-6, **0.83e-6**, 1.0e-6, 1.5e-6, 2.0e-6]                                            |

<h4>Amazon-Music</h4>

| Baseline | R@20   | N@20   | P@20   | Hyperparameter Search                                                                                                          |
|----------|--------|--------|--------|--------------------------------------------------------------------------------------------------------------------------------|
| MF-BPR   | 0.2868 | 0.1610 | 0.0303 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [1e-5, **1e-2**]                                                            |
| NGCF     | 0.2756 | 0.1528 | 0.0292 | lr: [0.0005, 0.001, 0.005, 0.01, 0.02, **0.03**, 0.04], l_w: [1e-5, 8e-3, **1e-2**]                                            |
| LightGCN | 0.3009 | 0.1664 | 0.0318 | lr: [0.0001, 0.0005, 0.001, 0.005, 0.01, **0.02**, 0.03, 0.04], l_w: [1e-5, **8e-3**, 1e-2]                                    |
| VBPR     | 0.2801 | 0.1590 | 0.0294 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**]                                                            |
| BM3      | 0.2958 | 0.1621 | 0.0309 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], reg_weight: [**0.1**, 0.01]                                                      |
| MGCN     | 0.2764 | 0.1602 | 0.0296 | lr: [**0.0001**, 0.001, 0.01], c_l: [0.001, 0.01, **0.1**]                                                                     |
| FREEDOM  | 0.3131 | 0.1769 | 0.0332 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [**1e-5**, 1e-2]                                                            |
| LATTICE* | 0.3215 | 0.2202 | 0.0338 | lr: [0.0001, 0.0005, **0.001**, 0.005, 0.01], regs: [1e-5, **1e-2**], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MICRO*   | 0.3308 | 0.2272 | 0.0345 | lr: [0.0001, 0.0005, **0.001**, 0.005, 0.01], regs: [1e-5, **1e-2**], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MMSSL*   | 0.3211 | 0.2168 | 0.0340 | lr generator: [0.00055**, **4.5e-4**, 5e-4, 5.4e-3, 5.6e-3], lr discriminator: [**2.5e-4**, 3e-4, 3.5e-4]                      |
| SGL      | 0.2969 | 0.1733 | 0.0321 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**]                                                            |

*: original implementation. The remaining models were run using [ELLIOT](https://github.com/sisinflab/Graph-Missing-Modalities).

**: 0.00055 has been paired only with 3e-4
