"""Fused Metal ray-marching kernels, compiled at runtime with `torch.mps.compile_shader`.

Volume and labelmap may be float32, float16 or bfloat16 (read in place and
promoted per sample; geometry and accumulation stay float32). One thread marches one ray: the sample grid and per-sample intensities
of the `grid_sample` path are never materialized. Semantics match `_fused.py`:
trilinear intensity sampling with zero padding, nearest-neighbor label routing
(round-half-even) and a constant per-ray step. The backward kernel computes
analytic gradients w.r.t. the camera-to-voxel transform, the ray endpoints, the
step size and (optionally) the volume.
"""

import functools

import torch

_HEADER = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline float fetch(device const T* v, int ix, int iy, int iz, int D, int H, int W) {
    if (ix < 0 || ix >= W || iy < 0 || iy >= H || iz < 0 || iz >= D) return 0.0f;
    return (float)v[(iz * H + iy) * W + ix];
}

struct Ray {
    float3 s, e, p0, dp;  // camera-space source and direction; voxel-space origin and direction
    int s_lo, s_hi;       // culled sample range
    bool hit;
};

// Load ray `gid` and cull its sample range to where the trilinear footprint can touch the volume
inline Ray setup_ray(
    device const float* M, device const float* src, device const float* tgt,
    int gid, int b, int n, int D, int H, int W, int S, int src_ray_stride, int src_batch_stride)
{
    Ray r;
    device const float* m = M + b * 12;
    int so = b * src_batch_stride + n * src_ray_stride;
    r.s = float3(src[so], src[so + 1], src[so + 2]);
    r.e = float3(tgt[gid * 3], tgt[gid * 3 + 1], tgt[gid * 3 + 2]) - r.s;
    float3 m0 = float3(m[0], m[1], m[2]), m1 = float3(m[4], m[5], m[6]), m2 = float3(m[8], m[9], m[10]);
    r.p0 = float3(dot(m0, r.s) + m[3], dot(m1, r.s) + m[7], dot(m2, r.s) + m[11]);
    r.dp = float3(dot(m0, r.e), dot(m1, r.e), dot(m2, r.e));

    float3 lo = float3(-2.0f), hi = float3(W + 1.0f, H + 1.0f, D + 1.0f);
    float3 safe = select(r.dp, float3(1e-12f), abs(r.dp) < 1e-12f);
    float3 ta = (lo - r.p0) / safe, tb = (hi - r.p0) / safe;
    float3 tmin = min(ta, tb), tmax = max(ta, tb);
    float t_lo = max(max(tmin.x, tmin.y), tmin.z);
    float t_hi = min(min(tmax.x, tmax.y), tmax.z);
    r.hit = t_lo <= t_hi;
    float inv_s = 1.0f / (float)(S - 1);
    r.s_lo = max((int)floor(t_lo / inv_s) - 2, 0);
    r.s_hi = min((int)ceil(t_hi / inv_s) + 3, S);
    return r;
}

"""

_KERNELS = """
kernel void raymarch_fwd(
    device const VT* vol     [[buffer(0)]],
    device const VT* lab     [[buffer(1)]],
    device const float* M    [[buffer(2)]],   // [B, 3, 4] camera -> voxel pixel coords
    device const float* src  [[buffer(3)]],   // [B, N or 1, 3]
    device const float* tgt  [[buffer(4)]],   // [B, N, 3]
    device const float* stp  [[buffer(5)]],   // [B, N]
    device float* out        [[buffer(6)]],   // [B, C, N], zero-initialized
    constant int& N          [[buffer(7)]],
    constant int& D          [[buffer(8)]],
    constant int& H          [[buffer(9)]],
    constant int& W          [[buffer(10)]],
    constant int& S          [[buffer(11)]],
    constant int& C          [[buffer(12)]],
    constant int& src_ray_stride   [[buffer(13)]],
    constant int& src_batch_stride [[buffer(14)]],
    uint tid [[thread_position_in_grid]])
{
    int gid = (int)tid;
    int b = gid / N, n = gid - b * N;
    Ray r = setup_ray(M, src, tgt, gid, b, n, D, H, W, S, src_ray_stride, src_batch_stride);
    if (!r.hit) return;

    float inv_s = 1.0f / (float)(S - 1);
    float acc1 = 0.0f;
    device float* o = out + (long)b * C * N + n;

    for (int k = r.s_lo; k < r.s_hi; ++k) {
        float3 p = r.p0 + ((float)k * inv_s) * r.dp;
        float3 f0 = floor(p);
        float3 f = p - f0;
        int ix = (int)f0.x, iy = (int)f0.y, iz = (int)f0.z;

        float c00 = mix(fetch(vol, ix, iy,     iz,     D, H, W), fetch(vol, ix + 1, iy,     iz,     D, H, W), f.x);
        float c01 = mix(fetch(vol, ix, iy + 1, iz,     D, H, W), fetch(vol, ix + 1, iy + 1, iz,     D, H, W), f.x);
        float c10 = mix(fetch(vol, ix, iy,     iz + 1, D, H, W), fetch(vol, ix + 1, iy,     iz + 1, D, H, W), f.x);
        float c11 = mix(fetch(vol, ix, iy + 1, iz + 1, D, H, W), fetch(vol, ix + 1, iy + 1, iz + 1, D, H, W), f.x);
        float val = mix(mix(c00, c01, f.y), mix(c10, c11, f.y), f.z);

        if (C == 1) {
            acc1 += val;
        } else {
            int cls = (int)fetch(lab, (int)rint(p.x), (int)rint(p.y), (int)rint(p.z), D, H, W);
            if (cls >= 0 && cls < C) o[(long)cls * N] += val;
        }
    }
    float step = stp[gid];
    if (C == 1) o[0] = acc1 * step;
    else for (int c = 0; c < C; ++c) o[(long)c * N] *= step;
}

kernel void raymarch_bwd(
    device const float* go   [[buffer(0)]],   // [B, C, N]
    device const VT* vol     [[buffer(1)]],
    device const VT* lab     [[buffer(2)]],
    device const float* M    [[buffer(3)]],
    device const float* src  [[buffer(4)]],
    device const float* tgt  [[buffer(5)]],
    device const float* stp  [[buffer(6)]],
    device float* gM         [[buffer(7)]],   // [B*N, 12] per-ray partials, reduced in torch
    device float* gsrc       [[buffer(8)]],   // [B*N, 3]
    device float* gtgt       [[buffer(9)]],   // [B*N, 3]
    device float* gstep      [[buffer(10)]],  // [B*N]
    device atomic_float* gvol [[buffer(11)]], // [D, H, W], zero-initialized
    constant int& N          [[buffer(12)]],
    constant int& D          [[buffer(13)]],
    constant int& H          [[buffer(14)]],
    constant int& W          [[buffer(15)]],
    constant int& S          [[buffer(16)]],
    constant int& C          [[buffer(17)]],
    constant int& src_ray_stride   [[buffer(18)]],
    constant int& src_batch_stride [[buffer(19)]],
    constant int& need_gvol  [[buffer(20)]],
    constant int& need_gcam  [[buffer(21)]],
    uint tid [[thread_position_in_grid]])
{
    int gid = (int)tid;
    int b = gid / N, n = gid - b * N;
    Ray r = setup_ray(M, src, tgt, gid, b, n, D, H, W, S, src_ray_stride, src_batch_stride);
    device const float* m = M + b * 12;
    float3 m0 = float3(m[0], m[1], m[2]), m1 = float3(m[4], m[5], m[6]), m2 = float3(m[8], m[9], m[10]);

    float step = stp[gid];
    float3 aM0 = 0.0f, aM1 = 0.0f, aM2 = 0.0f;  // dL/dM rows (3 weights each)
    float3 aT = 0.0f;                           // dL/dM translation column
    float3 gs = 0.0f, gt = 0.0f;
    float gst = 0.0f;
    float inv_s = 1.0f / (float)(S - 1);

    if (r.hit) for (int k = r.s_lo; k < r.s_hi; ++k) {
        float tt = (float)k * inv_s;
        float3 p = r.p0 + tt * r.dp;
        float3 f0 = floor(p);
        float3 f = p - f0;
        int ix = (int)f0.x, iy = (int)f0.y, iz = (int)f0.z;

        float v000 = fetch(vol, ix, iy, iz, D, H, W),         v001 = fetch(vol, ix + 1, iy, iz, D, H, W);
        float v010 = fetch(vol, ix, iy + 1, iz, D, H, W),     v011 = fetch(vol, ix + 1, iy + 1, iz, D, H, W);
        float v100 = fetch(vol, ix, iy, iz + 1, D, H, W),     v101 = fetch(vol, ix + 1, iy, iz + 1, D, H, W);
        float v110 = fetch(vol, ix, iy + 1, iz + 1, D, H, W), v111 = fetch(vol, ix + 1, iy + 1, iz + 1, D, H, W);

        float c00 = mix(v000, v001, f.x), c01 = mix(v010, v011, f.x);
        float c10 = mix(v100, v101, f.x), c11 = mix(v110, v111, f.x);
        float val = mix(mix(c00, c01, f.y), mix(c10, c11, f.y), f.z);

        int cls = 0;
        if (C > 1) cls = (int)fetch(lab, (int)rint(p.x), (int)rint(p.y), (int)rint(p.z), D, H, W);
        if (cls < 0 || cls >= C) continue;  // dropped in the forward pass
        float gg = go[((long)b * C + cls) * N + n];
        float g = gg * step;

        // Analytic trilinear derivatives w.r.t. the voxel-space position
        float3 d;
        d.x = ((v001 - v000) * (1 - f.y) + (v011 - v010) * f.y) * (1 - f.z)
            + ((v101 - v100) * (1 - f.y) + (v111 - v110) * f.y) * f.z;
        d.y = ((v010 - v000) * (1 - f.x) + (v011 - v001) * f.x) * (1 - f.z)
            + ((v110 - v100) * (1 - f.x) + (v111 - v101) * f.x) * f.z;
        d.z = ((v100 - v000) * (1 - f.x) + (v101 - v001) * f.x) * (1 - f.y)
            + ((v110 - v010) * (1 - f.x) + (v111 - v011) * f.x) * f.y;
        float3 gp = g * d;

        // dL/dM[i, j] = gpos_i * q_j, q = lerp(src, tgt, t) in camera space
        float3 q = r.s + tt * r.e;
        aM0 += gp.x * q; aM1 += gp.y * q; aM2 += gp.z * q;
        aT += gp;

        if (need_gcam) {
            float3 gq = m0 * gp.x + m1 * gp.y + m2 * gp.z;  // M[:, :3]^T @ gpos
            gs += (1.0f - tt) * gq;
            gt += tt * gq;
            gst += gg * val;
        }

        if (need_gvol) {
            for (int corner = 0; corner < 8; ++corner) {
                int cx = corner & 1, cy = (corner >> 1) & 1, cz = corner >> 2;
                int jx = ix + cx, jy = iy + cy, jz = iz + cz;
                if (jx < 0 || jx >= W || jy < 0 || jy >= H || jz < 0 || jz >= D) continue;
                float w = (cx ? f.x : 1 - f.x) * (cy ? f.y : 1 - f.y) * (cz ? f.z : 1 - f.z);
                atomic_fetch_add_explicit(&gvol[(jz * H + jy) * W + jx], g * w, memory_order_relaxed);
            }
        }
    }

    device float* gm = gM + (long)gid * 12;
    gm[0] = aM0.x; gm[1] = aM0.y; gm[2] = aM0.z; gm[3] = aT.x;
    gm[4] = aM1.x; gm[5] = aM1.y; gm[6] = aM1.z; gm[7] = aT.y;
    gm[8] = aM2.x; gm[9] = aM2.y; gm[10] = aM2.z; gm[11] = aT.z;
    if (need_gcam) {
        gsrc[gid * 3] = gs.x; gsrc[gid * 3 + 1] = gs.y; gsrc[gid * 3 + 2] = gs.z;
        gtgt[gid * 3] = gt.x; gtgt[gid * 3 + 1] = gt.y; gtgt[gid * 3 + 2] = gt.z;
        gstep[gid] = gst;
    }
}
"""

# Volume and labelmap element types the kernels are instantiated for; geometry stays float32
_DTYPES = {torch.float32: ("float", "f32"), torch.float16: ("half", "f16"), torch.bfloat16: ("bfloat", "bf16")}


def _build_source() -> str:
    """The shared helpers plus one copy of the kernels per volume dtype (`raymarch_fwd_f16`, ...)."""
    out = [_HEADER]
    for ctype, suffix in _DTYPES.values():
        body = _KERNELS.replace("raymarch_fwd", f"raymarch_fwd_{suffix}").replace(
            "raymarch_bwd", f"raymarch_bwd_{suffix}"
        )
        out.append(f"#define VT {ctype}\n{body}\n#undef VT\n")
    return "\n".join(out)


@functools.cache
def _library():
    return torch.mps.compile_shader(_build_source())


def _kernel(name: str, vol: torch.Tensor):
    """The `name` kernel instantiated for the volume's dtype."""
    if vol.dtype not in _DTYPES:
        raise TypeError(f"metal backend supports float32, float16 and bfloat16 volumes, got {vol.dtype}")
    return getattr(_library(), f"{name}_{_DTYPES[vol.dtype][1]}")


def _src_strides(src: torch.Tensor) -> tuple[int, int]:
    return (0 if src.shape[1] == 1 else 3), (0 if src.shape[0] == 1 else src.shape[1] * 3)


@torch.library.custom_op("nanodrr::metal_raymarch", mutates_args=())
def metal_raymarch(
    vol: torch.Tensor,
    lab: torch.Tensor,
    M: torch.Tensor,
    src: torch.Tensor,
    tgt: torch.Tensor,
    step: torch.Tensor,
    n_samples: int,
    n_classes: int,
    width: int,
) -> torch.Tensor:
    """Fused ray march on MPS; same contract as `_fused.fused_raymarch`.

    Args are dense on MPS: `vol`/`lab` `[D, H, W]` (float32/16 or bfloat16, same dtype), geometry float32: `M` `[B, 3, 4]`,
    `src` `[B, N, 3]` or `[B, 1, 3]`, `tgt` `[B, N, 3]`, `step` `[B, N]`.
    `width` is unused (kept for parity with the Triton op).
    """
    B, N, _ = tgt.shape
    D, H, W = vol.shape
    out = torch.zeros(B, n_classes, N, device=vol.device, dtype=torch.float32)
    _kernel("raymarch_fwd", vol)(
        vol, lab, M.contiguous(), src, tgt, step, out,
        N, D, H, W, n_samples, n_classes, *_src_strides(src),
        threads=B * N,
    )  # fmt: skip
    return out


@metal_raymarch.register_fake
def _(vol, lab, M, src, tgt, step, n_samples, n_classes, width):
    B, N, _ = tgt.shape
    return torch.empty(B, n_classes, N, device=vol.device, dtype=torch.float32)


def _setup_context(ctx, inputs, output):
    vol, lab, M, src, tgt, step, n_samples, n_classes, _width = inputs
    ctx.save_for_backward(vol, lab, M.contiguous(), src, tgt, step)
    ctx.n_samples = n_samples
    ctx.n_classes = n_classes


@torch.library.custom_op("nanodrr::metal_raymarch_bwd", mutates_args=())
def _metal_raymarch_bwd(
    go: torch.Tensor,
    vol: torch.Tensor,
    lab: torch.Tensor,
    M: torch.Tensor,
    src: torch.Tensor,
    tgt: torch.Tensor,
    step: torch.Tensor,
    n_samples: int,
    n_classes: int,
    need_gvol: bool,
    need_gcam: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward kernel launch: `(gvol, gM, gsrc, gtgt, gstep)`, unrequested ones empty."""
    if need_gvol and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "Volume gradients on the metal backend use nondeterministic atomics; "
            "disable deterministic algorithms or use backend='torch'"
        )
    B, N, _ = tgt.shape
    D, H, W = vol.shape
    dev, f32 = vol.device, torch.float32
    gM_part = torch.empty(B * N, 12, device=dev, dtype=f32)
    gvol = torch.zeros(D, H, W, device=dev, dtype=f32) if need_gvol else gM_part.new_empty(1)
    gsrc = torch.empty(B * N, 3, device=dev, dtype=f32) if need_gcam else gM_part.new_empty(1)
    gtgt = torch.empty(B * N, 3, device=dev, dtype=f32) if need_gcam else gM_part.new_empty(1)
    gstep = torch.empty(B * N, device=dev, dtype=f32) if need_gcam else gM_part.new_empty(1)

    _kernel("raymarch_bwd", vol)(
        go.contiguous(), vol, lab, M.contiguous(), src, tgt, step,
        gM_part, gsrc, gtgt, gstep, gvol,
        N, D, H, W, n_samples, n_classes, *_src_strides(src), int(need_gvol), int(need_gcam),
        threads=B * N,
    )  # fmt: skip
    # Per-ray partials are reduced here, so the pose gradient is deterministic
    gM = gM_part.view(B, N, 12).sum(dim=1).view(B, 3, 4)
    if not need_gvol:
        gvol = gvol.new_empty(0)
    if need_gcam:
        gsrc, gtgt, gstep = gsrc.view(B, N, 3), gtgt.view(B, N, 3), gstep.view(B, N)
    else:
        gsrc, gtgt, gstep = (gvol.new_empty(0) for _ in range(3))
    return gvol, gM, gsrc, gtgt, gstep


@_metal_raymarch_bwd.register_fake
def _(go, vol, lab, M, src, tgt, step, n_samples, n_classes, need_gvol, need_gcam):
    B, N, _ = tgt.shape
    f32 = torch.float32
    gM = torch.empty(B, 3, 4, device=M.device, dtype=f32)
    gvol = torch.empty(vol.shape, device=M.device, dtype=f32) if need_gvol else gM.new_empty(0)
    gsrc = torch.empty(B, N, 3, device=M.device, dtype=f32) if need_gcam else gM.new_empty(0)
    gtgt = torch.empty(B, N, 3, device=M.device, dtype=f32) if need_gcam else gM.new_empty(0)
    gstep = torch.empty(B, N, device=M.device, dtype=f32) if need_gcam else gM.new_empty(0)
    return gvol, gM, gsrc, gtgt, gstep


def _backward(ctx, go):
    vol, lab, M, src, tgt, step = ctx.saved_tensors
    need_gvol = ctx.needs_input_grad[0]
    need_gcam = any(ctx.needs_input_grad[i] for i in (3, 4, 5))
    gvol, gM, gsrc, gtgt, gstep = _metal_raymarch_bwd(
        go, vol, lab, M, src, tgt, step, ctx.n_samples, ctx.n_classes, need_gvol, need_gcam
    )
    if need_gcam:
        if src.shape[1] == 1:
            gsrc = gsrc.sum(dim=1, keepdim=True)
        if src.shape[0] == 1:
            gsrc = gsrc.sum(dim=0, keepdim=True)
    return (
        gvol.to(vol.dtype) if need_gvol else None,
        None,  # lab: nearest-neighbor lookup, not differentiable
        gM.to(M.dtype),
        gsrc.to(src.dtype) if need_gcam else None,
        gtgt.to(tgt.dtype) if need_gcam else None,
        gstep.to(step.dtype) if need_gcam else None,
        None,  # n_samples
        None,  # n_classes
        None,  # width
    )


torch.library.register_autograd("nanodrr::metal_raymarch", _backward, setup_context=_setup_context)
