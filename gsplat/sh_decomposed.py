from typing import Optional
import torch
from torch import Tensor
from gsplat.cuda._wrapper import _make_lazy_cuda_func


class _SphericalHarmonicsDecomposed(torch.autograd.Function):
    """Spherical Harmonics"""

    @staticmethod
    def forward(
        ctx, sh_degree: int, dirs: Tensor, dc: Tensor, coeffs: Tensor, masks: Tensor
    ) -> Tensor:
        colors = _make_lazy_cuda_func("compute_sh_decomposed_fwd")(sh_degree, dirs, dc, coeffs, masks)
        ctx.save_for_backward(dirs, dc, coeffs, masks)
        ctx.sh_degree = sh_degree
        ctx.num_bases = coeffs.shape[-2]
        return colors

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        dirs, dc, coeffs, masks = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        num_bases = ctx.num_bases
        compute_v_dirs = ctx.needs_input_grad[1]
        v_dc, v_coeffs, v_dirs = _make_lazy_cuda_func("compute_sh_decomposed_bwd")(
            num_bases,
            sh_degree,
            dirs,
            dc,
            coeffs,
            masks,
            v_colors.contiguous(),
            compute_v_dirs,
        )
        if not compute_v_dirs:
            v_dirs = None
        return None, v_dirs, v_dc, v_coeffs, None


def spherical_harmonics_decomposed(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    dc: Tensor,  # [..., 1, 3]
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,
) -> Tensor:
    """Computes spherical harmonics.

    Args:
        degrees_to_use: The degree to be used.
        dirs: Directions. [..., 3]
        dc: the Coefficients of the first degree. [..., 1, 3]
        coeffs: Coefficients. [..., K, 3]
        masks: Optional boolen masks to skip some computation. [...,] Default: None.

    Returns:
        Spherical harmonics. [..., 3]
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2] + 1, coeffs.shape
    assert dirs.shape[:-1] == dc.shape[:-2], (dirs.shape, dc.shape)
    assert dirs.shape[:-1] == coeffs.shape[:-2], (dirs.shape, coeffs.shape)
    assert dirs.shape[-1] == 3, dirs.shape
    assert dc.shape[-1] == 3, dc.shape
    assert coeffs.shape[-1] == 3, coeffs.shape
    if masks is not None:
        assert masks.shape == dirs.shape[:-1], masks.shape
        masks = masks.contiguous()
    return _SphericalHarmonicsDecomposed.apply(
        degrees_to_use, dirs.contiguous(), dc.contiguous(), coeffs.contiguous(), masks
    )
