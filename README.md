# Diverse Negative Sampling (DivNS) for Recommender Systems 

## Appendix
### Hyperparameter Settings

The following table lists the hyperparameter settings of DivNS with MF and LightGCN on our four datasets. The hyperparameters include the learning rate $lr$, the coefficient of $l_2$ regularisation, the cache ratio $r$, the size of the candidate set $m$ and the mixing coefficient $\lambda$.

| Model     | Dataset        | lr        | l₂        | r  | m  | λ   |
|-----------|----------------|-----------|-----------|----|----|-----|
| **MF**    | Amazon Beauty  | 1e-3      | 1e-4      | 4  | 10 | 0.7 |
|           | ML 1M          | 1e-3      | 1e-3      | 6  | 10 | 0.7 |
|           | Pinterest      | 1e-4      | 1e-3      | 4  | 10 | 0.5 |
|           | Yelp 2022      | 1e-3      | 1e-3      | 2  | 10 | 0.7 |
| **LightGCN** | Amazon Beauty | 1e-4   | 1e-4      | 4  | 10 | 0.7 |
|           | ML 1M          | 1e-3      | 1e-3      | 4  | 10 | 0.7 |
|           | Pinterest      | 1e-3      | 1e-3      | 4  | 10 | 0.5 |
|           | Yelp 2022      | 1e-3      | 1e-3      | 4  | 10 | 0.5 |

## File structure

The folder <code>modules</code> contains the implementation of two backbone recommenders -- MF and LightGCN. All negative samplers are implemented in each recommender file to be accustomed to the model. The code that generates the toy example is included in <code>selections/toy_example.py</code>. We also experimented with different diverse samplers in <code>sampler</code>.

## Reproduce results

Environment Requirement
The code has been tested running under Python 3.8. To install the required package, run the following command:
```
pip install -r requirements.txt
```

To run the code, use the following command 
```
python main.py --dataset=beauty --ns=divns --model=mf --r=5
```
