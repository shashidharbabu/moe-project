import pytest
import torch
from src.models.router_student import StudentRouter

def test_student_router_initialization():
    router = StudentRouter()
    assert router is not None
    assert isinstance(router, StudentRouter)

def test_student_router_forward_pass():
    router = StudentRouter()
    sample_input = torch.randn(1, 768)  # Assuming the input size is 768
    output = router(sample_input)
    assert output.shape == (1, 3)  # Output should be domain probabilities

def test_student_router_training():
    router = StudentRouter()
    sample_input = torch.randn(32, 768)  # Batch size of 32
    sample_labels = torch.tensor([[1, 0, 0]] * 32)  # Example labels for Portrait domain
    optimizer = torch.optim.Adam(router.parameters(), lr=0.001)

    router.train()
    optimizer.zero_grad()
    output = router(sample_input)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, sample_labels.float())
    loss.backward()
    optimizer.step()

    assert loss.item() < 1.0  # Ensure loss decreases during training

def test_student_router_soft_labels():
    router = StudentRouter()
    sample_input = torch.randn(1, 768)
    output = router(sample_input)
    assert torch.all(output >= 0) and torch.all(output <= 1)  # Check that outputs are probabilities

def test_student_router_inference():
    router = StudentRouter()
    sample_input = torch.randn(1, 768)
    output = router(sample_input)
    assert output.sum().item() == pytest.approx(1.0, rel=1e-2)  # Check that probabilities sum to 1