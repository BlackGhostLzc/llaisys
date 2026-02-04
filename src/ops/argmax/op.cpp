#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 1. Check Device Consistency
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // 2. Check Contiguity
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "ArgMax: all tensors must be contiguous.");

    // 3. Check Dtype
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "ArgMax: max_idx must be INT64.");

    // 4. Check Shapes and Calculate Dimensions
    ASSERT(vals->ndim() >= 1, "ArgMax: Input must be at least 1D.");
    
    size_t ndim = vals->ndim();
    size_t inner_size = vals->shape()[ndim - 1]; // The dimension to reduce (cols)
    size_t total_numel = vals->numel();
    size_t outer_size = total_numel / inner_size; // Batch * Seq ... (rows)

    // Output shapes should match input shape excluding the last dimension
    // OR output numel must equal outer_size (flattened check)
    ASSERT(max_idx->numel() == outer_size, "ArgMax: Output tensor size mismatch.");
    ASSERT(max_val->numel() == outer_size, "ArgMax: Output tensor size mismatch.");

    // 5. Dispatch
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            outer_size,
            inner_size
        );
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            outer_size,
            inner_size
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
