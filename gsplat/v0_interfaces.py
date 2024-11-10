import math
import torch
from typing import Tuple, Literal, Optional
from jaxtyping import Float, Int
from torch import Tensor
import gsplat.cuda._wrapper as wrapper
from .rasterize_to_visibilities import rasterize_to_visibilities


def get_intrinsics_matrix(fx, fy, cx, cy, device):
    K = torch.eye(3, device=device)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def project_gaussians(
    means3d: Float[Tensor, "*batch 3"],
    scales: Float[Tensor, "*batch 3"],
    glob_scale: float,
    quats: Float[Tensor, "*batch 4"],
    viewmat: Float[Tensor, "4 4"],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    block_width: int,
    clip_thresh: float = 0.01,
    filter_2d_kernel_size: float = 0.3,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in normalized quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       clip_thresh (float): minimum z depth threshold.
       filter_2d_kernel_size (float)

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **compensation** (Tensor): the density compensation for blurring 2D kernel
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
        - **cov3d** (Tensor): 3D covariances.
    """

    if glob_scale != 1.:
        scales = scales * glob_scale

    # 3x4 -> 4x4
    if viewmat.shape[0] == 3:
        viewmat = torch.concat(
            [
                viewmat,
                torch.tensor(
                    [[0., 0., 0., 1.]],
                    dtype=viewmat.dtype,
                    device=viewmat.device,
                ),
            ],
            dim=0,
        )

    radii, means2d, depths, conics, compensations = wrapper.fully_fused_projection(
        means=means3d,
        covars=None,
        quats=quats,
        scales=scales,
        viewmats=viewmat.unsqueeze(0),
        Ks=get_intrinsics_matrix(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            device=means3d.device,
        ).unsqueeze(0),
        width=img_width,
        height=img_height,
        eps2d=filter_2d_kernel_size,
        packed=False,
        calc_compensations=True,
        near_plane=clip_thresh,
    )

    tile_width = math.ceil(img_width / float(block_width))
    tile_height = math.ceil(img_height / float(block_width))
    tiles_per_gauss, isect_ids, flatten_ids = wrapper.isect_tiles(
        means2d,
        radii,
        depths,
        block_width,
        tile_width,
        tile_height,
        packed=False,
        n_cameras=1,
        camera_ids=None,
        gaussian_ids=None,
    )

    return means2d.squeeze(0), depths.squeeze(0), radii.squeeze(0), conics.squeeze(0), compensations.squeeze(0), tiles_per_gauss.squeeze(0), None


class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, D]
        opacities: Tensor,  # [C, N]
        backgrounds: Tensor,  # [C, D], Optional
        masks: Tensor,  # [C, tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, last_ids = wrapper._make_lazy_cuda_func(
            "rasterize_to_pixels_fwd"
        )(
            means2d.unsqueeze(0),
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = wrapper._make_lazy_cuda_func("rasterize_to_pixels_bwd")(
            means2d.unsqueeze(0),
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            absgrad,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs.squeeze(0)

        if ctx.needs_input_grad[4]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_to_pixels(
    means2d: Tensor,  # [N, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    colors: Tensor,  # [C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [C, N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    masks: Optional[Tensor] = None,  # [C, tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
    """

    C = isect_offsets.size(0)
    device = means2d.device
    assert packed is False
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(0)
        assert means2d.shape == (N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    render_colors, render_alphas = _RasterizeToPixels.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas


def rasterize_gaussians(
    xys: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
    block_width: int,
    background: Optional[Float[Tensor, "channels"]] = None,
    return_alpha: Optional[bool] = False,
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
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        block_width (int): MUST match whatever block width was used in the project_gaussians call. integer number of pixels between 2 and 16 inclusive
        background (Tensor): background color
        return_alpha (bool): whether to return alpha channel

    Returns:
        A Tensor:

        - **out_img** (Tensor): N-dimensional rendered output image.
        - **out_alpha** (Optional[Tensor]): Alpha channel of the rendered output image.
    """

    means2d = xys
    depths = depths.unsqueeze(0)
    radii = radii.unsqueeze(0)
    conics = conics.unsqueeze(0)
    colors = colors.unsqueeze(0)
    opacities = opacity.squeeze(-1).unsqueeze(0)
    backgrounds = background.unsqueeze(0)

    # Identify intersecting tiles
    tile_width = math.ceil(img_width / float(block_width))
    tile_height = math.ceil(img_height / float(block_width))
    tiles_per_gauss, isect_ids, flatten_ids = wrapper.isect_tiles(
        means2d.unsqueeze(0),
        radii,
        depths,
        block_width,
        tile_width,
        tile_height,
        packed=False,
        n_cameras=1,
        camera_ids=None,
        gaussian_ids=None,
    )
    isect_offsets = wrapper.isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        img_width,
        img_height,
        block_width,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=False,
        absgrad=True,
    )

    render_colors = render_colors.squeeze(0)
    render_alphas = render_alphas.squeeze(0).squeeze(-1)

    if return_alpha:
        return render_colors, render_alphas
    return render_colors


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
    means2d = xys.unsqueeze(0)
    depths = depths.unsqueeze(0)
    radii = radii.unsqueeze(0)
    conics = conics.unsqueeze(0)
    opacities = opacity.squeeze(-1).unsqueeze(0)

    # Identify intersecting tiles
    tile_width = math.ceil(img_width / float(block_width))
    tile_height = math.ceil(img_height / float(block_width))
    tiles_per_gauss, isect_ids, flatten_ids = wrapper.isect_tiles(
        means2d,
        radii,
        depths,
        block_width,
        tile_width,
        tile_height,
        packed=False,
        n_cameras=1,
        camera_ids=None,
        gaussian_ids=None,
    )
    isect_offsets = wrapper.isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    n_hit_pixels, opacity_scores, alpha_scores, visibility_scores = rasterize_to_visibilities(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        image_width=img_width,
        image_height=img_height,
        tile_size=block_width,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        masks=None,
        packed=False,
    )

    return n_hit_pixels.squeeze(0), opacity_scores.squeeze(0), alpha_scores.squeeze(0), visibility_scores.squeeze(0)


def spherical_harmonics(
    degrees_to_use: int,
    viewdirs: Float[Tensor, "*batch 3"],
    coeffs: Float[Tensor, "*batch D C"],
    method: Literal["poly", "fast"] = "fast",
) -> Float[Tensor, "*batch C"]:
    return wrapper.spherical_harmonics(
        degrees_to_use=degrees_to_use,
        dirs=viewdirs,
        coeffs=coeffs,
    )
