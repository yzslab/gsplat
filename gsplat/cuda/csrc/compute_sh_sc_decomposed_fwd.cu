#include "bindings.h"
#include "spherical_harmonics_mc_decomposed.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

template <typename T>
__global__ void compute_sh_sc_decomposed_fwd_kernel(
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const vec3<T> *__restrict__ dirs, // [N, 3]
    const T *__restrict__ dc,         // [N, 1, 1]
    const T *__restrict__ coeffs,     // [N, K, 1]
    const bool *__restrict__ masks,   // [N]
    T *__restrict__ colors            // [N, 1]
) {
    // parallelize over N
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }
    uint32_t elem_id = idx;
    uint32_t c = 0; // color channel
    if (masks != nullptr && !masks[elem_id]) {
        return;
    }
    sh_mc_coeffs_to_color_fast_decomposed<T, 1>(
        degrees_to_use,
        c,
        dirs[elem_id],
        dc + elem_id,
        coeffs + elem_id * K,
        colors + elem_id
    );
}

torch::Tensor compute_sh_sc_decomposed_fwd_tensor(
    const uint32_t degrees_to_use,
    const torch::Tensor &dirs,              // [..., 3]
    const torch::Tensor &dc,                // [..., 1, 1]
    const torch::Tensor &coeffs,            // [..., K, 1]
    const at::optional<torch::Tensor> masks // [...]
) {
    GSPLAT_DEVICE_GUARD(dirs);
    GSPLAT_CHECK_INPUT(dirs);
    GSPLAT_CHECK_INPUT(dc);
    GSPLAT_CHECK_INPUT(coeffs);
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(dc.size(-1) == 1, "dc must have last dimension 1");
    TORCH_CHECK(coeffs.size(-1) == 1, "coeffs must have last dimension 1");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;            // N_cameras * N_gaussians

    auto sizes = dirs.sizes().vec();
    sizes.back() = 1;
    torch::Tensor colors = torch::empty(sizes, dirs.options()); // [..., 1]

    // parallelize over N
    if (N) {
        compute_sh_sc_decomposed_fwd_kernel<float>
            <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS>>>(
                N,
                K,
                degrees_to_use,
                reinterpret_cast<vec3<float> *>(dirs.data_ptr<float>()),
                dc.data_ptr<float>(),
                coeffs.data_ptr<float>(),
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                colors.data_ptr<float>()
            );
    }
    return colors; // [..., 1]
}

} // namespace gsplat