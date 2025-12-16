import pytest
from src.inference.moe_mid_pipeline import MoEInferencePipeline

@pytest.fixture
def inference_pipeline():
    return MoEInferencePipeline()

def test_inference_portrait(inference_pipeline):
    prompt = "A high-fashion portrait"
    generated_image = inference_pipeline.generate_image(prompt)
    assert generated_image is not None
    # Additional assertions can be added to check image properties

def test_inference_food(inference_pipeline):
    prompt = "A delicious cheeseburger"
    generated_image = inference_pipeline.generate_image(prompt)
    assert generated_image is not None
    # Additional assertions can be added to check image properties

def test_inference_landscape(inference_pipeline):
    prompt = "A beautiful mountain range"
    generated_image = inference_pipeline.generate_image(prompt)
    assert generated_image is not None
    # Additional assertions can be added to check image properties

def test_inference_ambiguous_prompt(inference_pipeline):
    prompt = "A person eating at an outdoor cafe"
    generated_image = inference_pipeline.generate_image(prompt)
    assert generated_image is not None
    # Additional assertions can be added to check image properties

def test_inference_invalid_prompt(inference_pipeline):
    prompt = ""
    with pytest.raises(ValueError):
        inference_pipeline.generate_image(prompt)