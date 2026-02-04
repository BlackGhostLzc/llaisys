#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::byte *idx_out, std::byte *val_out, const std::byte *val_in,
            llaisysDataType_t dtype, size_t outer_size, size_t inner_size);

} // namespace llaisys::ops::cpu