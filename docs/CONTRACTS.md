# Day-0 contracts (frozen -- change only with P4's review)

| Topic | Rule |
|---|---|
| Seed | `42` everywhere (`ears_ids.common.config.SEED`) |
| Feature type | `float32` |
| Labels | `label_bin` (0 benign / 1 attack) and `label_multi` (index into `class_order.json`) |
| Class order | one file per dataset: `data/processed/{dataset}/class_order.json`; every member outputs probabilities in this order |
| Data | `data/processed/{dataset}/{train,val,test}.parquet` (paper split); `.../official/` for the official partition |
| Columns | readable names, kept through preprocessing |
| Artifacts | `artifacts/{dataset}/{member}/` e.g. `artifacts/nsl_kdd/skm_xgb/` |
| SKM-XGB files | `preprocessor.joblib`, `dae.pt` (P1), `selected_features.json` (names, not positions), `skm_train_{task}.parquet`, `xgb_{task}.json`, `meta_{task}.json` |
| Fit data | every `fit` / `fit_resample` sees the **training set only** |

`{task}` is `binary` or `multiclass`. `{member}` is the council member's `name` (`skm_xgb`).

## Module signatures (SKM-XGB phase)
```python
# ears_ids.common.data           (P2)
load_dataset(name, split="paper"|"official")        -> (train, val, test)
# ears_ids.council.members.skm_xgb
Preprocessor().fit(train); .transform(df)           -> DataFrame float32          # P2  (preprocessing.py)
DAESelector().fit(X, k=15); .selected_features_; .transform(X)                    # P1  (dae.py)
SKMSampler().fit_resample(X, y)                     -> (X_bal, y_bal)             # P1  (sampling.py)
train_xgb(X_bal, y_bal, X_val, y_val, task, seed)   -> (model, best_params)       # P3  (xgb_train.py)
SKMXGBDetector(task).fit(train, val) ...                                          # P4  (detector.py)
# ears_ids.common.metrics        (P3)
evaluate(model, X, y)                               -> dict incl. "far"
# ears_ids.explain               (P2)
explain_global(model, X); explain_local(model, x)
```

## Repo rules
1. One branch per module. 2. Nobody edits a file they don't own. 3. P4 reviews every merge.
4. Keep `pytest` green.
