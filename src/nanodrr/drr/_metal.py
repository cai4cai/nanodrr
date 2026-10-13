"""Fused Metal ray-marching kernel, compiled at runtime with `torch.mps.compile_shader`.

Forward only, float32. One thread marches one ray: the sample grid and
per-sample intensities of the `grid_sample` path are never materialized.
Semantics match `_fused.py`: trilinear intensity sampling with zero padding,
nearest-neighbor label routing (round-half-even) and a constant per-ray step.
"""

import functools

import torch

_SOURCE = """
#include <metal_stdlib>
using namespace metal;

inline float fetch(device const float* v, int ix, int iy, int iz, int D, int H, int W) {
    if (ix < 0 || ix >= W || iy < 0 || iy >= H || iz < 0 || iz >= D) return 0.0f;
    return v[(iz * H + iy) * W + ix];
}

kernel void raymarch_fwd(
    device const float* vol  [[buffer(0)]],
    device const float* lab  [[buffer(1)]],
    device const float* M    [[buffer(2)]],   // [B, 3, 4] camera -> voxel pixel coords
    device const float* src  [[buffer(3)]],   // [B or 1, N or 1, 3]
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
    uint gid [[thread_position_in_grid]])
{
    int B_N = (int)gid;
    int b = B_N / N;
    int n = B_N - b * N;

    device const float* m = M + b * 12;
    int so = b * src_batch_stride + n * src_ray_stride;
    float3 s = float3(src[so], src[so + 1], src[so + 2]);
    int to = B_N * 3;
    float3 t = float3(tgt[to], tgt[to + 1], tgt[to + 2]);
    float3 e = t - s;

    float3 p0 = float3(dot(float3(m[0], m[1], m[2]), s) + m[3],
                       dot(float3(m[4], m[5], m[6]), s) + m[7],
                       dot(float3(m[8], m[9], m[10]), s) + m[11]);
    float3 dp = float3(dot(float3(m[0], m[1], m[2]), e),
                       dot(float3(m[4], m[5], m[6]), e),
                       dot(float3(m[8], m[9], m[10]), e));

    // Cull to the samples whose trilinear footprint can touch the volume
    float3 lo = float3(-2.0f), hi = float3(W + 1.0f, H + 1.0f, D + 1.0f);
    float3 safe = select(dp, float3(1e-12f), abs(dp) < 1e-12f);
    float3 ta = (lo - p0) / safe, tb = (hi - p0) / safe;
    float3 tmin = min(ta, tb), tmax = max(ta, tb);
    float t_lo = max(max(tmin.x, tmin.y), tmin.z);
    float t_hi = min(min(tmax.x, tmax.y), tmax.z);
    if (!(t_lo <= t_hi)) return;

    float inv_s = 1.0f / (float)(S - 1);
    int s_lo = max((int)floor(t_lo / inv_s) - 2, 0);
    int s_hi = min((int)ceil(t_hi / inv_s) + 3, S);

    float acc1 = 0.0f;
    device float* o = out + (long)b * C * N + n;
    float step = stp[B_N];

    for (int k = s_lo; k < s_hi; ++k) {
        float tt = (float)k * inv_s;
        float3 p = p0 + tt * dp;
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
    if (C == 1) o[0] = acc1 * step;
    else for (int c = 0; c < C; ++c) o[(long)c * N] *= step;
}
"""


@functools.cache
def _library():
    return torch.mps.compile_shader(_SOURCE)


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
    """Fused ray march on MPS; same contract as `_fused.fused_raymarch` (forward only).

    Args are dense float32 on MPS: `vol`/`lab` `[D, H, W]`, `M` `[B, 3, 4]`,
    `src` `[B, N, 3]` or `[B, 1, 3]`, `tgt` `[B, N, 3]`, `step` `[B, N]`.
    """
    B, N, _ = tgt.shape
    D, H, W = vol.shape
    out = torch.zeros(B, n_classes, N, device=vol.device, dtype=torch.float32)
    _library().raymarch_fwd(
        vol,
        lab,
        M.contiguous(),
        src,
        tgt,
        step,
        out,
        N,
        D,
        H,
        W,
        n_samples,
        n_classes,
        0 if src.shape[1] == 1 else 3,
        0 if src.shape[0] == 1 else src.shape[1] * 3,
        threads=B * N,
    )
    return out
