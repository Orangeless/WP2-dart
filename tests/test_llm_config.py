import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from llm import chat


def test_models_config_is_mapping():
    assert isinstance(config.MODELS, dict)
    assert config.MODELS
    assert all(isinstance(tier, str) and isinstance(model, str) for tier, model in config.MODELS.items())


def test_chat_requires_model_and_provider_support():
    try:
        chat("hi", "openrouter/meta-llama/llama-3.2-3b-instruct", resolver="test", stage="test", inst_id="1", temperature=0.0, max_tokens=10, seed=1, log=[])
    except Exception as exc:
        assert "api key" in str(exc).lower() or "model" in str(exc).lower() or "provider" in str(exc).lower()
