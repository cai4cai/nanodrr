import pytest
import torch
from test_fused import (
    Parity,
    _assert_grads_match,
    _assert_render_parity,
    _render_both,
    make_camera,
    make_random_subject,
)

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


def _reduced(n_classes, dtype, device):
    """A subject with only the volume and labelmap stored in `dtype`, plus its float32 twin
    holding the same (quantized) values. Geometry stays float32, as `Subject.to` would not."""
    ref = make_random_subject(n_classes=n_classes)
    ref._image_hu = ref._image_hu.to(dtype).float()  # quantize, so both see identical samples
    low = make_random_subject(n_classes=n_classes)
    low._image_hu = ref._image_hu.to(dtype)
    low.label = low.label.to(dtype)
    return ref.to(device), low.to(device, dtype=None)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_classes", [1, 3])
def test_metal_reduced_precision_volume_matches_quantized_float32(dtype, n_classes):
    device = torch.device("mps")
    ref, low = _reduced(n_classes, dtype, device)
    k_inv, rt_inv, sdd, h, w = make_camera(device)

    assert low.image.dtype == dtype
    expected = render(ref, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")
    out = render(low, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")

    assert out.dtype == dtype
    # Same samples, same float32 accumulation: only the final rounding to `dtype` differs
    tol = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[dtype]
    assert ((expected - out.float()).abs().max() / expected.abs().max()).item() < tol


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metal_reduced_precision_volume_gradient(dtype):
    device = torch.device("mps")
    ref, low = _reduced(1, dtype, device)
    k_inv, rt_inv, sdd, h, w = make_camera(device)
    grads = []
    for subject in (ref, low):
        subject.convert_to_mu = False
        subject._image_hu.requires_grad_(True)
        render(subject, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal").float().sum().backward()
        grads.append(subject._image_hu.grad.float())

    assert low._image_hu.grad.dtype == dtype
    assert ((grads[0] - grads[1]).abs().max() / grads[0].abs().max()).item() < 2e-2


def test_metal_reduced_precision_halves_volume_footprint():
    """Reduced-precision volumes are read in place, with no float32 copy."""
    device = torch.device("mps")
    _, low = _reduced(1, torch.float16, device)
    k_inv, rt_inv, sdd, h, w = make_camera(device)
    before = torch.mps.current_allocated_memory()
    render(low, k_inv, rt_inv, sdd, h, w, n_samples=100, backend="metal")
    torch.mps.synchronize()

    assert torch.mps.current_allocated_memory() - before < low.image.numel() * 4


@pytest.mark.parametrize("orthographic", [False, True])
def test_metal_matches_torch_pose_gradients(parity, orthographic):
    _assert_grads_match(parity, "pose", orthographic=orthographic)


@pytest.mark.parametrize("n_classes", [1, 3])
def test_metal_matches_torch_volume_gradients(parity, n_classes):
    _assert_grads_match(parity, "volume", n_classes=n_classes)


def test_metal_matches_torch_intrinsics_gradients(parity):
    _assert_grads_match(parity, "intrinsics")


def test_metal_pose_gradients_deterministic():
    """Per-ray partials are reduced in torch, so pose gradients are bitwise stable."""
    subject, k_inv, rt_inv, sdd, h, w = _setup()
    grads = []
    for _ in range(2):
        rt = rt_inv.clone().requires_grad_(True)
        render(subject, k_inv, rt, sdd, h, w, n_samples=200, backend="metal").sum().backward()
        grads.append(rt.grad.clone())

    torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)


def test_metal_out_of_range_labels_have_finite_gradients():
    subject, k_inv, rt_inv, sdd, h, w = _setup(n_classes=5)
    subject.n_classes = 3
    rt = rt_inv.clone().requires_grad_(True)
    render(subject, k_inv, rt, sdd, h, w, n_samples=100, backend="metal").sum().backward()

    assert torch.isfinite(rt.grad).all()


def test_metal_batched_pose_gradients_match_torch(parity):
    """Batched poses against a shared detector reduce gradients over the broadcast dims."""
    from nanodrr.camera import make_rt_inv

    grads = []
    for backend, device in parity.runs():
        subject = make_random_subject().to(device)
        k_inv, _, sdd, h, w = make_camera(device)
        rt = make_rt_inv(
            torch.tensor([[5.0, -3.0, 8.0], [25.0, 4.0, -6.0]]),
            torch.tensor([[1.0, 80.0, -2.0], [-3.0, 78.0, 5.0]]),
            orientation="AP",
            isocenter=torch.zeros(3),
        ).to(device)
        rt.requires_grad_(True)
        render(subject, k_inv, rt, sdd, h, w, n_samples=200, backend=backend).sum().backward()
        grads.append(rt.grad[:, :3].cpu())

    ref, out = grads
    assert ((ref - out).abs().max() / ref.abs().max()).item() < 1e-3


def test_metal_backward_compiles():
    """`torch.compile` must trace both kernel launches as opaque custom ops."""
    subject, k_inv, rt_inv, sdd, h, w = _setup()

    def f(rt):
        return render(subject, k_inv, rt, sdd, h, w, n_samples=100, backend="metal")

    rt = rt_inv.clone().requires_grad_(True)
    f(rt).sum().backward()

    torch._dynamo.reset()
    rt_c = rt_inv.clone().requires_grad_(True)
    torch.compile(f, fullgraph=True)(rt_c).sum().backward()

    ref = rt.grad[0, :3]
    assert ((ref - rt_c.grad[0, :3]).abs().max() / ref.abs().max()).item() < 1e-3
