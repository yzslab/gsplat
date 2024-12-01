from typing import Optional, Tuple
import torch
from torch import Tensor
from gsplat.cuda._wrapper import _make_lazy_cuda_func

@torch.no_grad()
def isect_tiles_tile_based_culling(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    radii: Tensor,  # [C, N] or [nnz]
    depths: Tensor,  # [C, N] or [nnz]
    conics: Tensor,
    opacities: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    packed: bool = False,
    n_cameras: Optional[int] = None,
    camera_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        depths: Z-depth of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        conics: the upper triangular part of the inverted 2D covariance matrix. [C, N, 3] or [nnz, 3]
        opacities: [C, N] or [nnz]
        tile_size: Tile size.
        tile_width: Tile width.
        tile_height: Tile height.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        packed: If True, the input tensors are packed. Default: False.
        n_cameras: Number of cameras. Required if packed is True.
        camera_ids: The row indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [C, N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the camera and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [C * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.size()
        assert radii.shape == (nnz,), radii.size()
        assert depths.shape == (nnz,), depths.size()
        assert conics.shape == (nnz, 3), conics.size()
        assert opacities.shape == (nnz), opacities.size()
        assert camera_ids is not None, "camera_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_cameras is not None, "n_cameras is required if packed is True"
        camera_ids = camera_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        C = n_cameras

    else:
        C, N, _ = means2d.shape
        assert means2d.shape == (C, N, 2), means2d.size()
        assert radii.shape == (C, N), radii.size()
        assert depths.shape == (C, N), depths.size()
        assert conics.shape == (C, N, 3), conics.size()
        assert opacities.shape == (C, N), opacities.size()

    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("isect_tiles_tile_based_culling")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        camera_ids,
        gaussian_ids,
        C,
        tile_size,
        tile_width,
        tile_height,
        sort,
        True,  # DoubleBuffer: memory efficient radixsort
    )
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode_tile_based_culling(
    isect_ids: Tensor, n_cameras: int, tile_width: int, tile_height: int
) -> Tuple[Tensor, int]:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_cameras: Number of cameras.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [C, tile_height, tile_width]
        n_isects. [1]
    """
    offsets, n_isects = _make_lazy_cuda_func("isect_offset_encode_tile_based_culling")(
        isect_ids.contiguous(), n_cameras, tile_width, tile_height
    )
    return offsets, n_isects.item()