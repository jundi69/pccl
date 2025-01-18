#include "quantize_kernels.hpp"

#include <pccl_log.hpp>

#ifndef _MSC_VER
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline __forceinline
#endif

template<typename T>
struct minmax_t {
    T min;
    T max;
};


template<typename T>
FORCE_INLINE minmax_t<T> findMinAndMax(const T *__restrict__ src, const size_t count) {
    minmax_t<T> result{};
    result.min = std::numeric_limits<T>::max();
    result.max = std::numeric_limits<T>::min();
    for (size_t i = 0; i < count; ++i) {
        result.min = std::min(result.min, src[i]);
        result.max = std::max(result.max, src[i]);
    }
    return result;
}

using namespace ccoip::internal::quantize;

template<typename O, typename I, typename = std::enable_if_t<
    std::is_integral_v<O> && std::is_unsigned_v<O> && std::is_floating_point_v<I>> >
[[nodiscard]]
FORCE_INLINE static DeQuantizationMetaData minMaxQuant(O *__restrict__ dst, const I *__restrict__ src,
                                                       const size_t count) {
    minmax_t<I> minmax = findMinAndMax<I>(src, count);
    auto dif = static_cast<double>(minmax.max - minmax.min);
    if (dif == 0.0) {
        dif = 1.0;
    }
    for (size_t i = 0; i < count; ++i) {
        double relative = static_cast<double>(src[i] - minmax.min) / dif;
        dst[i] = static_cast<O>(relative * std::numeric_limits<I>::max());
    }
    return DeQuantizationMetaData::Make(minmax.min, minmax.max);
}

struct DtypeVariant {
    ccoip::ccoip_data_type_t dst_type;
    ccoip::ccoip_data_type_t src_type;

    friend bool operator==(const DtypeVariant &lhs, const DtypeVariant &rhs) {
        return lhs.dst_type == rhs.dst_type
               && lhs.src_type == rhs.src_type;
    }

    friend bool operator!=(const DtypeVariant &lhs, const DtypeVariant &rhs) {
        return !(lhs == rhs);
    }
};

template<>
struct std::hash<DtypeVariant> {
    std::size_t operator()(const DtypeVariant &pair) const noexcept {
        return std::hash<ccoip::ccoip_data_type_t>()(pair.dst_type)
               ^ std::hash<ccoip::ccoip_data_type_t>()(pair.src_type);
    }
};

typedef DeQuantizationMetaData (*quant_fn_t)(void *dst, const void *src, size_t count);

template<typename O, typename I, typename = std::enable_if_t<
    std::is_integral_v<O> && std::is_unsigned_v<O> && std::is_floating_point_v<I>> >
static constexpr quant_fn_t make_quantfn(DeQuantizationMetaData (*func)(O *, const I *, size_t)) {
    return reinterpret_cast<quant_fn_t>(func);
}


DeQuantizationMetaData ccoip::internal::quantize::performMinMaxQuantization(
    const std::span<std::byte> &dst_span,
    const std::span<const std::byte> &src_bytes,
    const ccoip_data_type_t quantized_type, const ccoip_data_type_t data_type) {
    static std::unordered_map<DtypeVariant, quant_fn_t> minmax_map = {
        // uint8 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
        {{ccoipFloat, ccoipUint8}, make_quantfn(&minMaxQuant<uint8_t, float>)},
        {{ccoipDouble, ccoipUint8}, make_quantfn(&minMaxQuant<uint8_t, double>)},

        // uint16 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
        {{ccoipFloat, ccoipUint16}, make_quantfn(&minMaxQuant<uint16_t, float>)},
        {{ccoipDouble, ccoipUint16}, make_quantfn(&minMaxQuant<uint16_t, double>)},

        // uint32 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
        {{ccoipFloat, ccoipUint32}, make_quantfn(&minMaxQuant<uint32_t, float>)},
        {{ccoipDouble, ccoipUint32}, make_quantfn(&minMaxQuant<uint32_t, double>)},

        // uint64 input with all floating point accumulation types (with fused de-quantization, hence only floating point dst types)
        {{ccoipFloat, ccoipUint64}, make_quantfn(&minMaxQuant<uint64_t, float>)},
        {{ccoipDouble, ccoipUint64}, make_quantfn(&minMaxQuant<uint64_t, double>)},
    };

    const DtypeVariant key{data_type, quantized_type};
    const auto it = minmax_map.find(key);
    if (it == minmax_map.end()) {
        LOG(FATAL) << "Unsupported data type combination: dst_type=" << key.dst_type
                << ", src_type=" << key.src_type
                << "; This likely means that nobody in their right minds would request this data type combination because there is no meaningful use case for it.";
        return DeQuantizationMetaData{};
    }

    const quant_fn_t fn = it->second;
    return fn(dst_span.data(), src_bytes.data(), src_bytes.size_bytes() / ccoip_data_type_size(data_type));
}
