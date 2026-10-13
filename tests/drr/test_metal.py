import pytest
import torch
from test_fused import Parity, _assert_render_parity, _render_both, make_camera, make_random_subject

from nanodrr.drr.renderer import render

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")


@pytest.fixture
def parity() -> Parity:
    """Metal on MPS against the torch reference on the CPU."""
    return Parity("metal", torch.device("mps"), "torch", torch.device("cpu"))


def _setup(n_classes=1, size=32):
    device = torch.device("mps")
    subject = make_random_subject(size=size, n_classes=n_classes).to(device)
    return (subject, *make_camera(device))


@pytest.mark.parametrize("orthographic", [False, True])
@pytest.mark.parametrize("n_classes", [1, 3])
def test_metal_matches_torch_forward(parity, orthographic, n_classes):
    ref, out = _render_both(parity, n_classes, orthographic)

    assert out.shape == ref.shape == (1, n_classes, 16, 16)
    _assert_render_parity(ref, out, n_classes)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("n_classes", [1, 3])
def test_metal_broadcasts_pose_batch(parity, n_classes):
    ref, out = _render_both(parity, n_classes, batched_pose=True)

    assert out.shape == ref.shape == (2, n_classes, 16, 16)
    _assert_render_parity(ref, out, n_classes)


def test_metal_multiclass_channels_sum_to_single_class():
    """Routing only splits the integral across channels; it must not change the total."""
    subject, k_inv, rt_inv, sdd, h, w = _setup(n_classes=3)
    multi = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")
    subject.n_classes = 1
    single = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")

    torch.testing.assert_close(multi.sum(dim=1, keepdim=True), single, rtol=1e-4, atol=1e-4)


def test_metal_out_of_range_labels_are_dropped():
    subject, k_inv, rt_inv, sdd, h, w = _setup(n_classes=5)
    subject.n_classes = 3  # fewer channels than the labelmap contains
    out = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")

    assert out.shape[1] == 3
    assert torch.isfinite(out).all()


def test_metal_ray_missing_volume_is_zero():
    """Culling must leave rays that never touch the volume exactly zero."""
    subject, k_inv, rt_inv, sdd, h, w = _setup()
    rt_inv = rt_inv.clone()
    rt_inv[:, :3, 3] += 10_000.0  # slide the whole geometry far off the volume
    out = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")

    assert (out == 0).all()


def test_metal_half_precision_subject_matches_float32():
    subject, k_inv, rt_inv, sdd, h, w = _setup()
    ref = render(subject, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")
    out = render(subject.half(), k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")

    assert out.shape == ref.shape
    assert ((ref - out.float()).abs().max() / ref.abs().max()).item() < 1e-2


def test_metal_rejects_gradients():
    subject, k_inv, rt_inv, sdd, h, w = _setup()

    with pytest.raises(NotImplementedError, match="backward"):
        render(subject, k_inv, rt_inv.clone().requires_grad_(True), sdd, h, w, n_samples=100, backend="metal")
