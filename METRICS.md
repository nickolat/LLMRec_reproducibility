## Hyperparameter study

In the following, the complete gridsearch on the chosen hyperparameters is reported.
In bold, the best hyperparameters.

### Netflix Dataset

| Baseline | R@10  | N@10   | R@20   | N@20   | R@50   | N@50   | P@20   | Hyperparameters                                                                                                                |
|----------|-------|--------|--------|--------|--------|--------|--------|--------------------------------------------------------------------------------------------------------------------------------|
| MF-BPR   | 0.0369 | 0.0182 | 0.0623 | 0.0246 | 0.1165 | 0.0352 | 0.0031 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**]                                                            |
| NGCF     | 0.0412 | 0.0207 | 0.0645 | 0.0265 | 0.1117 | 0.0358 | 0.0032 | lr: [0.0005, 0.001, 0.005, 0.01, **0.02**, 0.03, 0.04], l_w: [1e-5, 8e-3, **1e-2**]                                            |
| LightGCN | 0.0341 | 0.0162 | 0.0645 | 0.0239 | 0.1268 | 0.0362 | 0.0032 | lr: [0.0005, 0.001, 0.005, 0.01, **0.02**, 0.03, 0.04], l_w: [1e-5, **8e-3**, 1e-2]                                            |
| VBPR     | 0.0282 | 0.0121 | 0.0612 | 0.0205 | 0.1024 | 0.0287 | 0.0031 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**]                                                            |
| BM3      | 0.0374 | 0.0161 | 0.0726 | 0.0251 | 0.1295 | 0.0362 | 0.0036 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], reg_weight: [0.1, **0.01**]                                                      |
| MGCN     | 0.0401 | 0.0165 | 0.0645 | 0.0226 | 0.1111 | 0.0317 | 0.0032 | lr: [**0.0001**, 0.001, 0.01], c_l: [**0.001**, 0.01, 0.1]                                                                     |
| FREEDOM  | 0.0304 | 0.0137 | 0.0612 | 0.0214 | 0.0992 | 0.0288 | 0.0031 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [**1e-5**, 1e-2]                                                            |
| LATTICE* | 0.0423 | 0.0208 | 0.0656 | 0.0267 | 0.1100 | 0.0354 | 0.0033 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], regs: [1e-5, **1e-2**], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MICRO*   | 0.0260 | 0.0108 | 0.0602 | 0.0194 | 0.1230 | 0.0318 | 0.0030 | lr: [0.0001, **0.0005**, 0.001, 0.005, 0.01], regs: [**1e-5**, 1e-2], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MMSSL*   | 0.0325 | 0.0125 | 0.0667 | 0.0212 | 0.1165 | 0.0311 | 0.0033 | lr generator: [**0.00055****, 4.5e-4, 5e-4, 5.4e-3, 5.6e-3], lr discriminator: [2.5e-4, **3e-4**, 3.5e-4]                      |
| SGL      | 0.0444 | 0.0200 | 0.0732 | 0.0273 | 0.1182 | 0.0361 | 0.0037 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [1e-5, **1e-2**]                                                            |
| *RLMRec**   | 0.0336| 0.0144| 0.0683| 0.0231| 0.1220| 0.0336| 0.0034| layer_num: [1, 2, 3, **4**], reg_weight: [0.5e-6, **0.83e-6**, 1e-6, 1.5e-6, 2.0e-6]                                           |  

**Notes**:  
> \* original implementation. The remaining models were run using [ELLIOT](https://github.com/sisinflab/Graph-Missing-Modalities).

> \** 0.00055 has been paired only with 3e-4

#### LLMRec 
Note: to use the validation set instead of test set in the original implementations,
without additional codes, you have to replace `self.test(users_to_test, is_val=False)` with
`self.test(users_to_val, is_val=True)` in the `main.py` file.

##### Reproducibility

| LLM                   | Candidates | R@10   | N@10   | R@20   | N@20   | R@50   | N@50   | P@20   | Hyperparameters                                               |
|-----------------------|------------|--------|--------|--------|--------|--------|--------|--------|---------------------------------------------------------------|
| GPT-4-turbo           | LATTICE    | 0.0331 | 0.0190 | 0.0580 | 0.0253 | 0.1084 | 0.0352 | 0.0029 | lr: [5e-5, **1e-3**], temp: 0.6, top_p: 0.1, max_tokens: 1024 |
| GPT-4-turbo           | MMSSL      | 0.0287 | 0.0104 | 0.0547 | 0.0169 | 0.1079 | 0.0274 | 0.0027 | lr: [5e-5, **1e-3**], temp: 0.6, top_p: 0.1, max_tokens: 1024 |
| Llama-3.1-405B-Instruct | LATTICE    | 0.0304 | 0.0150 | 0.0461 | 0.0190 | 0.0997 | 0.0294 | 0.0023 | lr: [5e-5, **1e-3**], temp: 0.6, top_p: 0.1, max_tokens: 1024 |
| Llama-3.1-405B-Instruct| MMSSL      | 0.0136 | 0.0054 | 0.0287 | 0.0091 | 0.0818 | 0.0195 | 0.0014 | lr: [5e-5, **1e-3**], temp: 0.6, top_p: 0.1, max_tokens: 1024 |

##### Replicability

| Setup                   | LLM(s)                         | Candidates       | R@10  | N@10  | R@20  | N@20  | R@50  | N@50  | P@20  |
|-------------------------|--------------------------------|------------------|-------|-------|-------|-------|-------|-------|-------|
| Paper (original)        | gpt-3.5-turbo-0613, 16k        | LATTICE or MMSSL | 0.0531| 0.0272| 0.0829| 0.0347| 0.1382| 0.0456| 0.0041|
| Augmented Data Provided | gpt-3.5-turbo-0613, 16k        | Provided         | 0.0515| 0.0260| 0.0813| 0.0335| 0.1371| 0.0445| 0.0041|
| New Evaluation          | gpt-3.5-turbo-16k              | MMSSL            | 0.0114| 0.0039| 0.0347| 0.0096| 0.0791| 0.0185| 0.0017|
| New Evaluation          | gpt-3.5-turbo-16k              | LATTICE          | 0.0130| 0.0051| 0.0390| 0.0117| 0.0775| 0.0191| 0.0020|

> **Note**: “Candidates” refers to the Baseline used to generate the list of items from which the recommendation process starts

### Amazon-Music Dataset

| Baseline   | R@10   | N@10   | R@20   | N@20   | R@50   | N@50   | P@20   | Hyperparameters                                                                                                                |
|------------|--------|--------|--------|--------|--------|--------|--------|--------------------------------------------------------------------------------------------------------------------------------|
| MF-BPR     | 0.2086 | 0.1375 | 0.2868 | 0.1610 | 0.4088 | 0.1908 | 0.0303 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [1e-5, **1e-2**]                                                            |
| NGCF       | 0.1937 | 0.1281 | 0.2756 | 0.1528 | 0.4086 | 0.1852 | 0.0292 | lr: [0.0005, 0.001, 0.005, 0.01, 0.02, **0.03**, 0.04], l_w: [1e-5, 8e-3, **1e-2**]                                            |
| LightGCN   | 0.2148 | 0.1405 | 0.3009 | 0.1664 | 0.4535 | 0.2032 | 0.0318 | lr: [0.0001, 0.0005, 0.001, 0.005, 0.01, **0.02**, 0.03, 0.04], l_w: [1e-5, **8e-3**, 1e-2]                                    |
| VBPR       | 0.2041 | 0.1361 | 0.2801 | 0.1590 | 0.4020 | 0.1887 | 0.0294 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**]                                                            |
| BM3        | 0.2094 | 0.1359 | 0.2958 | 0.1621 | 0.4306 | 0.1947 | 0.0309 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], reg_weight: [**0.1**, 0.01]                                                      |
| MGCN       | 0.2032 | 0.1381 | 0.2764 | 0.1602 | 0.3991 | 0.1901 | 0.0296 | lr: [**0.0001**, 0.001, 0.01], c_l: [0.001, 0.01, **0.1**]                                                                     |
| FREEDOM    | 0.2282 | 0.1513 | 0.3131 | 0.1769 | 0.4598 | 0.2128 | 0.0332 | lr: [0.0001, 0.0005, 0.001, **0.005**, 0.01], l_w: [**1e-5**, 1e-2]                                                            |
| LATTICE*   | 0.2321 | 0.1855 | 0.3215 | 0.2202 | 0.4605 | 0.2638 | 0.0338 | lr: [0.0001, 0.0005, **0.001**, 0.005, 0.01], regs: [1e-5, **1e-2**], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MICRO*     | 0.2419 | 0.1931 | 0.3308 | 0.2272 | 0.4663 | 0.2700 | 0.0345 | lr: [0.0001, 0.0005, **0.001**, 0.005, 0.01], regs: [1e-5, **1e-2**], n_layers: 1, n_ui_layers: 2, topk: 20, lambda_coeff: 0.7 |
| MMSSL*     | 0.2315 | 0.1822 | 0.3211 | 0.2168 | 0.4684 | 0.2628 | 0.0340 | lr generator: [0.00055**, **4.5e-4**, 5e-4, 5.4e-3, 5.6e-3], lr discriminator: [**2.5e-4**, 3e-4, 3.5e-4]                      |
| SGL        | 0.2203 | 0.1498 | 0.2969 | 0.1733 | 0.4111 | 0.2017 | 0.0321 | lr: [0.0001, 0.0005, 0.001, 0.005, **0.01**], l_w: [1e-5, **1e-2**]                                                            |

**Notes**:  
> \* original implementation. The remaining models were run using [ELLIOT](https://github.com/sisinflab/Graph-Missing-Modalities)

> \** 0.00055 has been paired only with 3e-4

#### LLMRec
Note: to use the validation set instead of test set in the original implementations,
without additional codes, you have to replace `self.test(users_to_test, is_val=False)` with
`self.test(users_to_val, is_val=True)` in the `main.py` file.

| LLM | Candidates | R@10   | N@10   | R@20   | N@20   | R@50   | N@50   | P@20   | Hyperparameters                                                               |
|-----|------------|--------|--------|--------|--------|--------|--------|--------|-------------------------------------------------------------------------------|
| gpt-3.5-turbo-16k    | LATTICE    | 0.1837 | 0.1458 | 0.2632 | 0.1771 | 0.4096 | 0.2237 | 0.0280 | lr: [5e−5, 1e−3, 2.5e−4, **9.5e−4**], temp: 0.6, top_p: 0.1, max_tokens: 1024 |
| gpt-3.5-turbo-16k    | MMSSL      | 0.1858 | 0.1502 | 0.2667 | 0.1817 | 0.4064 | 0.2266 | 0.0283 | lr: [5e−5, 1e−3, 2.5e−4, **9.5e−4**], temp: 0.6, top_p: 0.1, max_tokens: 1024 |

> **Note**: “Candidates” refers to the Baseline used to generate the list of items from which the recommendation process starts
