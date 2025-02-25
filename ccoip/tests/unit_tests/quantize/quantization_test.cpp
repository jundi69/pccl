#include <gtest/gtest.h>
#include <quantize.hpp>
#include <quantize_kernels.hpp>

TEST(QuantizationTest, TestPerfectInverse) {
    float values[] = {0.3, 0.4, 0.5, 0.6, 0.7};

    uint8_t quantized_values[5] = {};

    std::span dst_span(reinterpret_cast<std::byte *>(quantized_values), sizeof(quantized_values));
    std::span src_span(reinterpret_cast<const std::byte *>(values), sizeof(values));

    auto meta = ccoip::internal::quantize::performQuantization(dst_span, src_span, ccoip::ccoipQuantizationMinMax, ccoip::ccoipUint8, ccoip::ccoipFloat);

    float min_value = *reinterpret_cast<float *>(meta.min_value.data());
    float max_value = *reinterpret_cast<float *>(meta.max_value.data());
    float dequantized_values[5] = {};
    for (int i = 0; i < 5; ++i) {
        dequantized_values[i] = ccoip::internal::quantize::deQuantizeMinMaxScalar(quantized_values[i], min_value, max_value);
    }

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
