"""Python bindings for custom Cuda functions"""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects


def hit_pixel_count(
    xys: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    opacity: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
    block_width: int,
) -> Tensor:
    """Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        block_width (int): MUST match whatever block width was used in the project_gaussians call. integer number of pixels between 2 and 16 inclusive

    Returns:
        hit_pixel_count, important_opacity_score, important_alpha_score, important_visibility_score
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    return _CountGaussianHitPixels.apply(
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        opacity.contiguous(),
        img_height,
        img_width,
        block_width,
    )


class _CountGaussianHitPixels(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        radii: Float[Tensor, "*batch 1"],
        conics: Float[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        opacity: Float[Tensor, "*batch 1"],
        img_height: int,
        img_width: int,
        block_width: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        num_points = xys.size(0)
        tile_bounds = (
            (img_width + block_width - 1) // block_width,
            (img_height + block_width - 1) // block_width,
            1,
        )
        block = (block_width, block_width, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            hit_pixel_count = torch.zeros((xys.shape[0]), dtype=torch.int, device=xys.device)
            important_opacity_score = torch.zeros((xys.shape[0]), dtype=torch.float, device=xys.device)
            important_alpha_score = torch.zeros((xys.shape[0]), dtype=torch.float, device=xys.device)
            important_visibility_score = torch.zeros((xys.shape[0]), dtype=torch.float, device=xys.device)
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                depths,
                radii,
                cum_tiles_hit,
                tile_bounds,
                block_width,
            )

            hit_pixel_count, important_opacity_score, important_alpha_score, important_visibility_score = _C.hit_pixel_count_forward(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                opacity,
            )

        return hit_pixel_count, important_opacity_score, important_alpha_score, important_visibility_score

    @staticmethod
    def backward(ctx, count, score):
        return (
            None,  # xys
            None,  # depths
            None,  # radii
            None,  # conics
            None,  # num_tiles_hit
            None,  # opacity
            None,  # img_height
            None,  # img_width
            None,  # block_width
        )
