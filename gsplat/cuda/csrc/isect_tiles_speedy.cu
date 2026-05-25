/**
 * Precise Tile Intersect proposed by Speedy-Splat: https://speedysplat.github.io/
 * Some code snippets are copied from:
 *   https://github.com/j-alex-hanson/speedy-splat-rasterizer/blob/b38abb6da7a7217999f8c5ec0e9219284464ffe0/cuda_rasterizer/auxiliary.h
 *   https://github.com/nerfstudio-project/gsplat/commit/3d4f9027f36d70ac8e89536e14c247de377f74b5
 */

#include "bindings.h"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

__device__ inline float2 accutile_ellipse_intersection(
    float A, float B, float C, float disc, float t, float2 p,
    bool isY, float coord
) {
    float p_u   = isY ? p.y : p.x;
    float p_v   = isY ? p.x : p.y;
    float coeff = isY ? A : C;

    float h         = coord - p_u;
    float sqrt_term = sqrtf(disc * h * h + t * coeff);

    return {(-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v};
}

__device__ inline uint32_t accutile_process_tiles(
    float A, float B, float C, float disc, float t, float2 p,
    float2 bbox_min, float2 bbox_max, float2 bbox_argmin, float2 bbox_argmax,
    int2 rect_min, int2 rect_max,
    uint32_t tile_size, uint32_t tile_width, bool isY,
    int64_t iid_enc, uint32_t tile_n_bits, int64_t depth_id_enc,
    uint32_t flatten_idx, int64_t *isect_ids, int32_t *flatten_ids,
    int64_t *cur_idx
) {
    float BLOCK = (float)tile_size;

    if (isY) {
        rect_min    = {rect_min.y, rect_min.x};
        rect_max    = {rect_max.y, rect_max.x};
        bbox_min    = {bbox_min.y, bbox_min.x};
        bbox_max    = {bbox_max.y, bbox_max.x};
        bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    intersect_max_line = {bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK;
    if (bbox_min.x <= min_line) {
        intersect_min_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, min_line);
    } else {
        intersect_min_line = intersect_max_line;
    }

#pragma unroll 1
    for (int u = rect_min.x; u < rect_max.x; ++u) {
        max_line = min_line + BLOCK;
        if (max_line <= bbox_max.x) {
            intersect_max_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, max_line);
        }

        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
            ellipse_min = bbox_min.y;
        } else {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
            ellipse_max = bbox_max.y;
        } else {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        int min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / BLOCK)));
        int max_tile_v = min(rect_max.y, max(rect_min.y, (int)(ellipse_max / BLOCK + 1)));

        tiles_count += max_tile_v - min_tile_v;

        if (isect_ids != nullptr) {
#pragma unroll 1
            for (int v = min_tile_v; v < max_tile_v; v++) {
                int64_t tile_id       = isY ? (int64_t)(u * tile_width + v) : (int64_t)(v * tile_width + u);
                isect_ids[*cur_idx]   = iid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[*cur_idx] = static_cast<int32_t>(flatten_idx);
                ++(*cur_idx);
            }
        }

        intersect_min_line = intersect_max_line;
        min_line           = max_line;
    }
    return tiles_count;
}

/****************************************************************************
 * Gaussian Tile Intersection
 ****************************************************************************/

template <typename T>
__global__ void isect_tiles_speedy(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t N_C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const T *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const T *__restrict__ depths,                    // [C, N] or [nnz]
    const T *__restrict__ conics,                    // [C, N, 3] or [nnz, 3]
    const T *__restrict__ opacities,                 // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : N_C * N)) {
        return;
    }

    const OpT radius = radii[idx];
    if (radius <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    // SnugBox

    const float A = conics[idx * 3];     // con_o.x
    const float B = conics[idx * 3 + 1]; // con_o.y
    const float C = conics[idx * 3 + 2]; // con_o.z

    float2 mean2d = {(float)means2d[2 * idx], (float)means2d[2 * idx + 1]};

    // Calculate discriminant
    float disc = B * B - A * C;

    // If ill-formed ellipse, return
    if (A <= 0 || C <= 0 || disc >= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    // Threshold: opacity * Gaussian = 1 / 255
    float t = 2.0f * log(opacities[idx] * 255.0f);

    float x_term = sqrt(-(B * B * t) / (disc * A));
    x_term = (B < 0) ? x_term : -x_term;
    float y_term = sqrt(-(B * B * t) / (disc * C));
    y_term = (B < 0) ? y_term : -y_term;

    float2 bbox_argmin = { mean2d.y - y_term, mean2d.x - x_term };
    float2 bbox_argmax = { mean2d.y + y_term, mean2d.x + x_term };

    float2 bbox_min = {
      accutile_ellipse_intersection(A, B, C, disc, t, mean2d, true, bbox_argmin.x).x,
      accutile_ellipse_intersection(A, B, C, disc, t, mean2d, false, bbox_argmin.y).x
    };
    float2 bbox_max = {
      accutile_ellipse_intersection(A, B, C, disc, t, mean2d, true, bbox_argmax.x).y,
      accutile_ellipse_intersection(A, B, C, disc, t, mean2d, false, bbox_argmax.y).y
    };

    // Rectangular tile extent of ellipse
    float tile_size_f = (float)tile_size;
    int2 rect_min = {
        max(0, min((int)tile_width, (int)(bbox_min.x / tile_size_f))),
        max(0, min((int)tile_height, (int)(bbox_min.y / tile_size_f)))
    };
    int2 rect_max = {
        max(0, min((int)tile_width, (int)(bbox_max.x / tile_size_f + 1.f))),
        max(0, min((int)tile_height, (int)(bbox_max.y / tile_size_f + 1.f)))
    };

    int y_span = rect_max.y - rect_min.y;
    int x_span = rect_max.x - rect_min.x;

    // If no tiles are touched, return
    if (y_span * x_span == 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    bool isY = y_span < x_span;

    int64_t cur_idx = first_pass ? 0 : ((idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1]);

    int64_t cid_enc = 0;
    int64_t depth_id_enc = 0;
    if (!first_pass) {
        int64_t cid; // camera id
        if (packed) {
            // parallelize over nnz
            cid = camera_ids[idx];
            // gid = gaussian_ids[idx];
        } else {
            // parallelize over C * N
            cid = idx / N;
            // gid = idx % N;
        }
        cid_enc = cid << (32 + tile_n_bits);
        depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    }

    uint32_t count = accutile_process_tiles(
        A, B, C, disc, t, mean2d,
        bbox_min, bbox_max, bbox_argmin, bbox_argmax,
        rect_min, rect_max,
        tile_size, tile_width, isY,
        cid_enc, tile_n_bits, depth_id_enc, idx,
        first_pass ? nullptr : isect_ids,
        first_pass ? nullptr : flatten_ids,
        &cur_idx
    );

    if (first_pass) {
        tiles_per_gauss[idx] = static_cast<int32_t>(count);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_speedy_tensor(
    const torch::Tensor &means2d,                    // [C, N, 2] or [nnz, 2]
    const torch::Tensor &radii,                      // [C, N] or [nnz]
    const torch::Tensor &depths,                     // [C, N] or [nnz]
    const torch::Tensor &conics,                     // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                  // [C, N] or [nnz]
    const at::optional<torch::Tensor> &camera_ids,   // [nnz]
    const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool double_buffer
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(depths);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(opacities);
    if (camera_ids.has_value()) {
        GSPLAT_CHECK_INPUT(camera_ids.value());
    }
    if (gaussian_ids.has_value()) {
        GSPLAT_CHECK_INPUT(gaussian_ids.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t N = 0, nnz = 0, total_elems = 0;
    int64_t *camera_ids_ptr = nullptr;
    int64_t *gaussian_ids_ptr = nullptr;
    if (packed) {
        nnz = means2d.size(0);
        total_elems = nnz;
        TORCH_CHECK(
            camera_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, camera_ids and gaussian_ids must be provided."
        );
        camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
        gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
    } else {
        N = means2d.size(1); // number of gaussians
        total_elems = C * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    torch::Tensor tiles_per_gauss =
        torch::empty_like(depths, depths.options().dtype(torch::kInt32));

    int64_t n_isects;
    torch::Tensor cum_tiles_per_gauss;
    if (total_elems) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_speedy_total_elems",
            [&]() {
                isect_tiles_speedy<<<
                    (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    reinterpret_cast<scalar_t *>(conics.data_ptr<scalar_t>()),
                    reinterpret_cast<scalar_t *>(opacities.data_ptr<scalar_t>()),
                    nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    tiles_per_gauss.data_ptr<int32_t>(),
                    nullptr,
                    nullptr
                );
            }
        );
        cum_tiles_per_gauss = torch::cumsum(tiles_per_gauss.view({-1}), 0);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    torch::Tensor isect_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt32));
    if (n_isects) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_speedy_n_isects",
            [&]() {
                isect_tiles_speedy<<<
                    (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    reinterpret_cast<scalar_t *>(conics.data_ptr<scalar_t>()),
                    reinterpret_cast<scalar_t *>(opacities.data_ptr<scalar_t>()),
                    cum_tiles_per_gauss.data_ptr<int64_t>(),
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    nullptr,
                    isect_ids.data_ptr<int64_t>(),
                    flatten_ids.data_ptr<int32_t>()
                );
            }
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
        torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>()
            );
            cub::DoubleBuffer<int32_t> d_values(
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>()
            );
            GSPLAT_CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                d_keys,
                d_values,
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
            switch (d_keys.selector) {
            case 0: // sorted items are stored in isect_ids
                isect_ids_sorted = isect_ids;
                break;
            case 1: // sorted items are stored in isect_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n",
            // d_values.selector);
        } else {
            GSPLAT_CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>(),
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
        }
        return std::make_tuple(
            tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted
        );
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

} // namespace gsplat