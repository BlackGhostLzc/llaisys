#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>


template <typename T>
void argmax_(int64_t *idx_out, T *val_out, const T *val_in, 
             size_t outer_size, size_t inner_size) {
    
    // Iterate over the "batch" dimension (flattened outer dims)
    // #pragma omp parallel for
    for (size_t i = 0; i < outer_size; ++i) {
        
        // Pointer to the start of the current row
        const T* row_ptr = val_in + i * inner_size;

        // Initialize max tracking
        float max_val_f = -std::numeric_limits<float>::infinity();
        int64_t max_index = 0;
        
        // If inner_size is 0 (empty tensor), handled by loop condition, but theoretically standard dictates checks.
        // Assuming inner_size >= 1 per tensor spec.

        // Initialize with the first element to handle -inf inputs correctly
        if (inner_size > 0) {
             if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                max_val_f = llaisys::utils::cast<float>(row_ptr[0]);
             } else {
                max_val_f = static_cast<float>(row_ptr[0]);
             }
             max_index = 0;
        }

        // Iterate through the reduction dimension
        for (size_t j = 1; j < inner_size; ++j) {
            float current_val_f = 0.0f;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                current_val_f = llaisys::utils::cast<float>(row_ptr[j]);
            } else {
                current_val_f = static_cast<float>(row_ptr[j]);
            }

            if (current_val_f > max_val_f) {
                max_val_f = current_val_f;
                max_index = static_cast<int64_t>(j);
            }
        }

        // Write results
        idx_out[i] = max_index;
        
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            val_out[i] = llaisys::utils::cast<T>(max_val_f);
        } else {
            val_out[i] = static_cast<T>(max_val_f);
        }
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *idx_out, std::byte *val_out, const std::byte *val_in,
            llaisysDataType_t dtype, size_t outer_size, size_t inner_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_(
            reinterpret_cast<int64_t *>(idx_out),
            reinterpret_cast<float *>(val_out),
            reinterpret_cast<const float *>(val_in),
            outer_size, inner_size);
    case LLAISYS_DTYPE_BF16:
        return argmax_(
            reinterpret_cast<int64_t *>(idx_out),
            reinterpret_cast<llaisys::bf16_t *>(val_out),
            reinterpret_cast<const llaisys::bf16_t *>(val_in),
            outer_size, inner_size);
    case LLAISYS_DTYPE_F16:
        return argmax_(
            reinterpret_cast<int64_t *>(idx_out),
            reinterpret_cast<llaisys::fp16_t *>(val_out),
            reinterpret_cast<const llaisys::fp16_t *>(val_in),
            outer_size, inner_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu