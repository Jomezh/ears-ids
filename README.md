# EARS-IDS

Explainable, Adversarially Robust, Self-Testing Intrusion Detection System (final-year S7-S8 project).
Layout and rules: `docs/ARCHITECTURE.md`. Team agreements: `docs/CONTRACTS.md`.

## Start here (works today on fake data)
```bash
pip install -e ".[dev]"
python scripts/make_fixture.py       # fake dataset -> data/processed/fixture/
python scripts/run_pipeline.py       # chain runs end to end, saves artifacts/fixture/skm_xgb/
pytest                                # 15 tests
```
Every module has a working placeholder, so the chain already runs. Replace the *inside* of your files, keep the
function signatures, keep `pytest` green.

## Who owns what (SKM-XGB phase)
| Files | Owner | State now |
|---|---|---|
| `common/data.py`, `council/members/skm_xgb/preprocessing.py`, `explain/shap_explain.py` | P2 | loader works on fixture; scaler only; SHAP = stubs |
| `council/members/skm_xgb/dae.py`, `sampling.py` | P1 | DAE = placeholder (first k columns); sampler = first cut |
| `council/members/skm_xgb/xgb_train.py`, `common/metrics.py` | P3 | fixed params, no search; metrics working |
| `common/interface.py`, `common/config.py`, `council/`, `scripts/`, `tests/`, `docs/` | P4 | done for the fixture |
| `adversarial/`, `response/`, `selftest/` | later | empty, reserved |
