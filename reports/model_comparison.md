# Model comparison

Generated 2026-11-14T07:15:37.089065+00:00 by `src/evaluation/evaluate.py`; best value per column in **bold** (lowest for RMSE/MAE, highest otherwise).

## Protocol
* Temporal split at 1998-03-07T02:21:09+00:00; relevant = test rating >= 4.
* Ranking: 98 warm test users (training history + >= 1 relevant test item); items rated in training excluded; candidates = 1616 training items.
* coverage / diversity / novelty at K=10; RMSE / MAE on 2,949 test ratings of warm users.
* 97.5% of the warm users' relevant test items exist in the training catalogue (upper bound on recall).

## Warm users
| model | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | map@10 | coverage | diversity | novelty | rmse | mae | precision@5 | recall@5 | ndcg@5 | hit_rate@5 | map@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.1082** | 0.0830 | 0.1340 | **0.4592** | 0.0760 | 0.0470 | **0.7467** | 1.2539 | 1.0309 | 0.8327 | **0.1163** | 0.0617 | 0.1344 | 0.3469 | 0.0896 |
| item_knn | 0.1061 | 0.0566 | 0.1190 | 0.3776 | 0.0739 | **0.1132** | 0.7084 | 2.0419 | 1.0743 | 0.8267 | 0.1102 | 0.0264 | 0.1190 | 0.2857 | 0.0841 |
| mf_svd | 0.0776 | 0.0855 | 0.1140 | 0.3673 | 0.0685 | 0.1021 | 0.7318 | 2.9404 | **0.9706** | **0.7636** | 0.0939 | 0.0556 | 0.1133 | 0.2857 | 0.0790 |
| hybrid | 0.0939 | **0.1010** | **0.1533** | 0.4490 | **0.0978** | 0.1126 | 0.4581 | **3.1163** | **0.9706** | **0.7636** | 0.1143 | **0.0742** | **0.1585** | **0.3571** | **0.1138** |

Primary metric: `ndcg@10` - best model: **hybrid**.

## Cold-start slice (popularity fallback)
192 of 301 test users have no training history; 192 of them have a relevant test item. Every model serves them its popularity fallback:

| model | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | map@10 |
|---|---:|---:|---:|---:|---:|
| popularity | 0.2917 | 0.0865 | 0.3153 | 0.9219 | 0.1804 |
| item_knn | 0.2917 | 0.0865 | 0.3153 | 0.9219 | 0.1804 |
| mf_svd | 0.2917 | 0.0865 | 0.3153 | 0.9219 | 0.1804 |
| hybrid | 0.2917 | 0.0865 | 0.3153 | 0.9219 | 0.1804 |
