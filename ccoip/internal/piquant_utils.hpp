#pragma once

#include <thread>

#include <piquant.hpp>

namespace ccoip::internal {
    inline piquant::context &get_quant_ctx() {
        static piquant::context s_ctx{std::max(1u, std::thread::hardware_concurrency())};
        return s_ctx;
    }

    [[nodiscard]] inline piquant::dtype get_piquant_dtype(const ccoip_data_type_t type) {
        switch (type) {
            case ccoipInt8:
                return piquant::dtype::int8;
            case ccoipUint8:
                return piquant::dtype::uint8;
            case ccoipInt16:
                return piquant::dtype::int16;
            case ccoipUint16:
                return piquant::dtype::uint16;
            case ccoipInt32:
                return piquant::dtype::int32;
            case ccoipUint32:
                return piquant::dtype::uint32;
            case ccoipInt64:
                return piquant::dtype::int64;
            case ccoipUint64:
                return piquant::dtype::uint64;
            case ccoipFloat:
                return piquant::dtype::f32;
            case ccoipDouble:
                return piquant::dtype::f64;
            default:
                break;
        }
        throw std::logic_error{"Unsupported data type"};
    }
}
