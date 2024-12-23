#include "reduce_kernels.hpp"

#include <pccl_log.hpp>

template<typename T>
void performSumReduce(T *dst, const T *src, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] += src[i];
    }
}

void ccoip::performReduction(std::span<std::byte> &span, const std::span<const std::byte> &data,
                             const ccoip_data_type_t data_type, const ccoip_reduce_op_t op) {
    switch (op) {
        case ccoipOpSum: {
            switch (data_type) {
                case ccoipInt8: {
                    const size_t count = span.size_bytes() / sizeof(int8_t);
                    performSumReduce(reinterpret_cast<int8_t *>(span.data()),
                                     reinterpret_cast<const int8_t *>(data.data()), count);
                    break;
                }
                case ccoipUint8: {
                    const size_t count = span.size_bytes() / sizeof(uint8_t);
                    performSumReduce(reinterpret_cast<uint8_t *>(span.data()),
                                     reinterpret_cast<const uint8_t *>(data.data()), count);
                    break;
                }
                case ccoipInt16: {
                    const size_t count = span.size_bytes() / sizeof(int16_t);
                    performSumReduce(reinterpret_cast<int16_t *>(span.data()),
                                     reinterpret_cast<const int16_t *>(data.data()), count);
                    break;
                }
                case ccoipUint16: {
                    const size_t count = span.size_bytes() / sizeof(uint16_t);
                    performSumReduce(reinterpret_cast<uint16_t *>(span.data()),
                                     reinterpret_cast<const uint16_t *>(data.data()), count);
                    break;
                }
                case ccoipInt32: {
                    const size_t count = span.size_bytes() / sizeof(int32_t);
                    performSumReduce(reinterpret_cast<int32_t *>(span.data()),
                                     reinterpret_cast<const int32_t *>(data.data()), count);
                    break;
                }
                case ccoipUint32: {
                    const size_t count = span.size_bytes() / sizeof(uint32_t);
                    performSumReduce(reinterpret_cast<uint32_t *>(span.data()),
                                     reinterpret_cast<const uint32_t *>(data.data()), count);
                    break;
                }
                case ccoipInt64: {
                    const size_t count = span.size_bytes() / sizeof(int64_t);
                    performSumReduce(reinterpret_cast<int64_t *>(span.data()),
                                     reinterpret_cast<const int64_t *>(data.data()), count);
                    break;
                }
                case ccoipUint64: {
                    const size_t count = span.size_bytes() / sizeof(uint64_t);
                    performSumReduce(reinterpret_cast<uint64_t *>(span.data()),
                                     reinterpret_cast<const uint64_t *>(data.data()), count);
                    break;
                }
                case ccoipFloat: {
                    const size_t count = span.size_bytes() / sizeof(float);
                    performSumReduce(reinterpret_cast<float *>(span.data()),
                                     reinterpret_cast<const float *>(data.data()), count);
                    break;
                }
                case ccoipDouble: {
                    const size_t count = span.size_bytes() / sizeof(double);
                    performSumReduce(reinterpret_cast<double *>(span.data()),
                                     reinterpret_cast<const double *>(data.data()), count);
                    break;
                }
                default: {
                    LOG(ERR) << "Unsupported data type: " << data_type;
                    break;
                }
            }
            break;
        }
        default: {
            LOG(FATAL) << "TODO: NOT IMPLEMENTED";
            break;
        }
    }
}
