import pytest
import torch
from zipformer import ZipformerBlock, Zipformer, ZipformerConvModule


def test_zipformer_block():
    """Test ZipformerBlock with basic functionality"""
    block = ZipformerBlock(
        dim=512,
        heads=8,
        mult=4
    )
    
    x = torch.randn(32, 100, 512)
    output = block(x)
    
    assert output.shape == (32, 100, 512), f"Expected shape (32, 100, 512), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_zipformer_conv_module():
    """Test ZipformerConvModule"""
    module = ZipformerConvModule(
        dim=512,
        causal=False,
        expansion_factor=2,
        kernel_size=31,
        dropout=0.1
    )
    
    x = torch.randn(16, 50, 512)
    output = module(x)
    
    assert output.shape == (16, 50, 512), f"Expected shape (16, 50, 512), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_zipformer():
    """Test Zipformer model"""
    zipformer = Zipformer()
    
    x = torch.randn(4, 100, 80)
    output = zipformer(x)
    
    assert output.shape[0] == 4, f"Expected batch size 4, got {output.shape[0]}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_zipformer_block_different_sizes():
    """Test ZipformerBlock with different dimensions"""
    configs = [
        (256, 4, 2),
        (512, 8, 4),
        (768, 12, 4)
    ]
    
    for dim, heads, mult in configs:
        block = ZipformerBlock(dim=dim, heads=heads, mult=mult)
        x = torch.randn(8, 50, dim)
        output = block(x)
        
        assert output.shape == (8, 50, dim), f"Expected shape (8, 50, {dim}), got {output.shape}"
        assert not torch.isnan(output).any(), f"Output contains NaN for dim={dim}"


def test_zipformer_gradient_flow():
    """Test that gradients flow through ZipformerBlock"""
    block = ZipformerBlock(dim=256, heads=4, mult=2)
    x = torch.randn(4, 20, 256, requires_grad=True)
    
    output = block(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients are None"
    assert not torch.isnan(x.grad).any(), "Gradients contain NaN"
    assert torch.abs(x.grad).sum() > 0, "Gradients are zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
