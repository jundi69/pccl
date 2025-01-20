#pragma once

#include <ccoip_client_state.hpp>
#include <ccoip_types.hpp>
#include <span>
#include <tinysockets.hpp>

namespace ccoip::reduce {
    /// Performs a pipeline ring reduce operation.
    void pipelineRingReduce(
        CCoIPClientState &client_state,
        uint64_t tag, const std::span<const std::byte> &src_buf, const std::span<std::byte> &dst_buf,
        ccoip_data_type_t data_type, ccoip_data_type_t quantized_type,
        ccoip_reduce_op_t op, ccoip_quantization_algorithm_t quantization_algorithm, size_t rank,
        size_t world_size,
        const std::vector<ccoip_uuid_t> &ring_order,
        const std::unordered_map<ccoip_uuid_t, std::unique_ptr<tinysockets::BlockingIOSocket>> &
        peer_tx_sockets,
        const std::unordered_map<ccoip_uuid_t, std::unique_ptr<tinysockets::BlockingIOSocket>> &
        peer_rx_sockets);
}; // namespace ccoip::reduce
