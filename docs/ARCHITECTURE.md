# Architecture (how the repo is organised and why)

## The five layers = five folders
```
src/ears_ids/
  common/       shared by everyone: config, data loading, metrics, the Detector contract
  council/      LAYER 1  members/skm_xgb/ (built)   members/<next>/ (later)   voting.py
  explain/      LAYER 2  SHAP/LIME                  (P2)
  adversarial/  LAYER 3  attacks + hardening        (P3, empty now)
  response/     LAYER 4  confidence-tiered actions  (empty now)
  selftest/     LAYER 5  attack replay + drift log  (P4/Jose, empty now)
```
A layer may use layers below it and `common/`, never the reverse. That is what lets the team stop at any layer
without breaking the ones underneath.

## The one rule that keeps it extensible
Every council member subclasses `common/interface.py::Detector` and offers the same calls:
`fit`, `predict_proba`, `predict`, `predict_one`, `save`, `load`.
Layers 2-5 only use those calls, so they never care what is inside a member.

## Adding a council member later (e.g. deep model, autoencoder)
1. Create `council/members/<name>/` with a `detector.py`.
2. Subclass `Detector`, set `name = "<name>"`, implement `fit / predict_proba / save / load`.
3. Do your own preprocessing inside the member. Return probabilities in `class_order`.
4. Copy `tests/test_interface.py` for your member. If those pass, the council can use it.

## Who uses what
- Layer 2 (explain): the member's model + readable feature names.
- Layer 3 (adversarial): `predict_proba` to attack and to measure robustness.
- Layer 4 (response): `predict_one(...)["confidence"]` to choose auto-act vs notify.
- Layer 5 (selftest): `predict_proba` / `predict_one` on replayed attacks; logs detection rate over time.
