from copy import deepcopy
from typing import Any, Dict, List, Tuple

from .abmil import ABMILGatedBaseConfig, ABMILModel
from .acmil import ACMILModel
from .clam_sb import CLAMSBModel
from .dsmil import DSMILModel
from .transmil import TransMILModel


MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "abmil": {
        "gate": True,
        "embed_dim": 512,
        "attn_dim": 384,
        "num_fc_layers": 1,
        "dropout": 0.25,
    },
    "clam_sb": {
        "gate": True,
        "embed_dim": 512,
        "attn_dim": 384,
        "num_fc_layers": 1,
        "dropout": 0.25,
    },
    "dsmil": {
        "embed_dim": 512,
        "num_fc_layers": 1,
        "dropout": 0.25,
        "temperature": 1.0,
    },
    "transmil": {
        "embed_dim": 512,
        "n_heads": 8,
        "n_layers": 2,
        "ffn_dim": 1024,
        "dropout": 0.1,
        "max_tokens": 2048,
    },
    "acmil": {
        "embed_dim": 512,
        "num_fc_layers": 1,
        "dropout": 0.25,
        "num_branches": 4,
        "topk_ratio": 0.1,
    },
}


MODEL_CAPABILITIES: Dict[str, Dict[str, bool]] = {
    "abmil": {
        "supports_full_attention": True,
    },
    "clam_sb": {
        "supports_full_attention": True,
    },
    "dsmil": {
        "supports_full_attention": True,
    },
    "transmil": {
        # TransMIL may truncate very long bags before self-attention.
        "supports_full_attention": False,
    },
    "acmil": {
        "supports_full_attention": True,
    },
}


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower().replace("-", "_")
    aliases = {
        "clam": "clam_sb",
    }
    return aliases.get(normalized, normalized)


def get_available_models() -> List[str]:
    return sorted(MODEL_DEFAULTS.keys())


def get_default_model_kwargs(model_name: str) -> Dict[str, Any]:
    name = normalize_model_name(model_name)
    if name not in MODEL_DEFAULTS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available models: {', '.join(get_available_models())}"
        )
    return deepcopy(MODEL_DEFAULTS[name])


def get_model_capabilities(model_name: str) -> Dict[str, bool]:
    name = normalize_model_name(model_name)
    if name not in MODEL_CAPABILITIES:
        return {}
    return deepcopy(MODEL_CAPABILITIES[name])


def create_mil_model(
    model_name: str,
    in_dim: int = 1536,
    num_classes: int = 2,
    **overrides,
) -> Tuple[Any, Dict[str, Any], Dict[str, bool]]:
    name = normalize_model_name(model_name)
    defaults = get_default_model_kwargs(name)

    for key, value in overrides.items():
        if value is None:
            continue
        if key in defaults:
            defaults[key] = value

    model_kwargs = {"in_dim": in_dim, "num_classes": num_classes, **defaults}

    if name == "abmil":
        config = ABMILGatedBaseConfig(**model_kwargs)
        model = ABMILModel(config)
    elif name == "clam_sb":
        model = CLAMSBModel(**model_kwargs)
    elif name == "dsmil":
        model = DSMILModel(**model_kwargs)
    elif name == "transmil":
        model = TransMILModel(**model_kwargs)
    elif name == "acmil":
        model = ACMILModel(**model_kwargs)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available models: {', '.join(get_available_models())}"
        )

    capabilities = get_model_capabilities(name)
    return model, model_kwargs, capabilities
