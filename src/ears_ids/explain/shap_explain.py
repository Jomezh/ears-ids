"""OWNER: P2.  STUBS -- SHAP work (week 3). Interface is frozen; fill in the bodies.

explain_global: global + per-class SHAP plots; compare with the paper's Fig. 6.
explain_local : one flow -> per-feature contributions (this is what Layer 2 shows per alert).
Both receive the XGBoost model and DataFrames with readable feature names.
"""


def explain_global(model, X, out_dir=None):
    raise NotImplementedError("P2: implement global/per-class SHAP")


def explain_local(model, x_row, out_dir=None):
    raise NotImplementedError("P2: implement local SHAP explanation for one flow")
