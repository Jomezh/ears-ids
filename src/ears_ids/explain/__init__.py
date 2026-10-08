"""LAYER 2 -- Explainability (SHAP/LIME per alert, council agreement views). Owner: P2."""

from .shap_explain import explain_global, explain_local

__all__ = ["explain_global", "explain_local"]
