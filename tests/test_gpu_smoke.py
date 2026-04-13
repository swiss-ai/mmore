import pytest
import torch


@pytest.mark.gpu
def test_cuda_basic_tensor_ops():
    """Smoke test that requires a real GPU runner (checks CUDA and simple tensor ops)."""
    assert torch.cuda.is_available(), "CUDA must be available on GPU runners"
    # create a tensor on GPU, do a simple op and validate results
    x = torch.tensor([1.0, 2.0], device="cuda")
    y = x * 2.0
    assert y.device.type == "cuda"
    assert torch.allclose(y.cpu(), torch.tensor([2.0, 4.0]))
