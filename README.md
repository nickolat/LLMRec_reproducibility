# How Powerful are LLMs to Support Multimodal Recommendation? A Reproducibility Study of LLMRec

This repository contains reproducibility and benchmarking codes for the [LLMRec paper](https://doi.org/10.1145/3616855.3635853).

-----------

## Datasets

The anonymous data storage is available [here](https://drive.google.com/file/d/1ktu5GOBoL0uUrdM70EQVXHZpMc3LRctB/view?usp=sharing).

`Netflix` [Original Split]: For replicability, dataset is available in [LLMRec GitHub repository](https://github.com/HKUDS/LLMRec.git).
For reproducibility, check the [data storage](https://drive.google.com/file/d/1ktu5GOBoL0uUrdM70EQVXHZpMc3LRctB/view?usp=sharing) (Files contained in ./data/Netflix/Train_Test have been created following the authors' pipeline).  

`Netflix` [Our Split]: For our benchmarking we used data available in the [data storage]((https://drive.google.com/file/d/1ktu5GOBoL0uUrdM70EQVXHZpMc3LRctB/view?usp=sharing)) (path: ./data/Netflix/Train_Val_Test) 

`Amazon-DigitalMusic`: The original dataset is available [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html), and processed with [Ducho](https://github.com/sisinflab/Ducho.git): 
1. Downloading of the Original Dataset (Digital_Music) via Ducho/demos/demo_recsys/download_amazon.sh
2. Processing of the dataset via Ducho/demos/demo_recsys/prepare_dataset.py with name='Digital_Music' and the meta dataset including also 'title' (two checks on its value should be added: NaN values or values with a string length = 0 are not allowed)

The already processed dataset is available in the [data storage](https://drive.google.com/file/d/1ktu5GOBoL0uUrdM70EQVXHZpMc3LRctB/view?usp=sharing) (path: ./data/Amazon).

To run each experiment, you have to put all the necessary input data in:
```
├─ LLMRec/
    ├── data/
        ├── netflix/
        ├── amazon/
```

## Usage

### RQ1: LLMRec replicability and reproducibility

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

### RQ1: Baselines reproducibility

For LATTICE and MMSSL use the official LLMRec repository;
for MICRO use its official [repository](https://github.com/CRIPAC-DIG/MICRO).

For the other baselines, use the last version of [ELLIOT](https://github.com/sisinflab/Graph-Missing-Modalities) repository: 
- Use the corresponding configuration files in the `config_files` directory of the repository. 
- Add `binarize: True` in the config files in order to use the provided versions of both datasets. 
- To enable deterministic behavior with CUDA, set: `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`.


### RQ2: Benchmarking with LLama

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

### RQ2: Benchmarking with GPT-4 Turbo

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

### RQ3: Benchmarking with new baselines and RLMRec

For the new recommendation baselines, follow the same instructions defined [above](#rq1-baselines-reproducibility).

For RLMRec, use the official [repository](https://github.com/HKUDS/RLMRec). 
- Substitute the corresponding directories in RLMRec (./emb, ./item, ./user) to the ones in the downloaded repository (path: ./generation). 
- Dataset already augmented and processed is available in the [data storage](https://drive.google.com/file/d/1ktu5GOBoL0uUrdM70EQVXHZpMc3LRctB/view?usp=sharing) (path: ./data/Netflix/Train_Val_Test/RLMRec).


### RQ4: Benchmarking on Amazon-music dataset

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

### RQ5: Topological properties of the LLM-augmented user-item graph

The code for the computation of the following characteristics: `['space_size', 'shape', 'density', 'gini_user',
                            'gini_item', 'average_degree_users', 'average_degree_items',
                            'average_clustering_coefficient_dot_users',
                            'average_clustering_coefficient_dot_items', 'degree_assortativity_users',
                            'degree_assortativity_items']` is contained in the Topology directory and based on the [GitHub repository](https://github.com/sisinflab/Topology-Graph-Collaborative-Filtering.git)

```
python Topology/check_dataset.py
python Topology/generate_only_characteristics.py
```
The already processed and analyzed data are available in the [data storage](https://drive.google.com/file/d/1ktu5GOBoL0uUrdM70EQVXHZpMc3LRctB/view?usp=sharing) (path: ./data_Topology)
