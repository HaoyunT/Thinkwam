import torch

from thinkwam_p0.models.sigreg import SIGReg


def test_sigreg_is_finite_differentiable_and_penalizes_collapse():
    torch.manual_seed(7)
    regularizer = SIGReg(knots=17, num_projections=64)
    regularizer.train()
    embeddings = torch.randn(3, 256, 16, requires_grad=True)
    gaussian_loss = regularizer(embeddings)
    collapsed_loss = regularizer(torch.zeros_like(embeddings))
    assert torch.isfinite(gaussian_loss)
    assert torch.isfinite(collapsed_loss)
    assert collapsed_loss > gaussian_loss
    gaussian_loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_sigreg_eval_does_not_change_random_directions():
    regularizer = SIGReg(knots=9, num_projections=16).eval()
    embeddings = torch.randn(2, 32, 8)
    first = regularizer(embeddings)
    second = regularizer(embeddings)
    assert torch.equal(first, second)
