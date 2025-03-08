#include "quantize.hpp"

#include <ccoip_types.hpp>
#include <pccl_log.hpp>
#include <quantize_kernels.hpp>
#include <span>


ccoip::internal::quantize::DeQuantizationMetaData ccoip::internal::quantize::performQuantization(const std::span<std::byte> &dst_span, const std::span<const std::byte> &src_span,
                                                                             const ccoip_quantization_algorithm_t quantization_algorithm,
                                                                             const ccoip_data_type_t quantized_type, const ccoip_data_type_t data_type) {

    switch (quantization_algorithm) {
        case ccoipQuantizationNone: {
            LOG(BUG) << "performQuantization should never be called with ccoipQuantizationNone.";
            return {};
        }
        case ccoipQuantizationMinMax: {
            return performMinMaxQuantization(dst_span, src_span, quantized_type, data_type);
        }
        case ccoipQuantizationZeroPointScale: {
            return performZeroPointScaleQuantization(dst_span, src_span, quantized_type, data_type);
        }
        default: {
            LOG(BUG) << "Unsupported quantization algorithm: " << quantization_algorithm;
            return {};
        }
    }
}

void ccoip::internal::quantize::performQuantizationAndDequantization(const std::span<std::byte> &dst_span,
        const std::span<const std::byte> &src_span, const ccoip_quantization_algorithm_t quantization_algorithm,
        const ccoip_data_type_t quantized_type, const ccoip_data_type_t data_type) {
    switch (quantization_algorithm) {
        case ccoipQuantizationNone: {
            LOG(BUG) << "performQuantization should never be called with ccoipQuantizationNone.";
            return;
        }
        case ccoipQuantizationMinMax: {
            performMinMaxQuantizationAndDequantization(dst_span, src_span, quantized_type, data_type);
            return;
        }
        default: {
            LOG(BUG) << "Unsupported quantization algorithm: " << quantization_algorithm;
        }
    }
}
