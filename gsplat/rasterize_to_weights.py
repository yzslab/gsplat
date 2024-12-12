from typing import Optional
import torch
from torch import Tensor
from .cuda._wrapper import _make_lazy_cuda_func


def rasterize_to_weights(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    opacities: Tensor,  # [C, N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    pixel_weights: Tensor,  # [C, H, W],
    masks: Optional[torch.Tensor] = None,
    packed: bool = False,
):
    C = isect_offsets.size(0)

    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert opacities.shape == (C, N), opacities.shape

    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    accum_weights, reverse_count, blend_weights, dist_accum = _make_lazy_cuda_func(
        "rasterize_to_weights"
    )(
        means2d,
        conics,
        opacities,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        pixel_weights,
    )

    return accum_weights, reverse_count, blend_weights, dist_accum
