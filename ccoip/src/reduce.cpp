#include "reduce.hpp"

#include <cassert>
#include <ccoip_master_state.hpp>
#include <cstring>
#include <ccoip_packets.hpp>
#include <ccoip_types.hpp>
#include <quantize.hpp>
#include <reduce_kernels.hpp>
#include <tinysockets.hpp>

namespace {

    /**
     * \brief Run the "reduce-scatter" stage of the ring in a single pipeline step.
     *
     * - Sends the `tx_span` (possibly quantized data) to the next rank in the ring.
     * - Receives the chunk from the previous rank in the ring into `recv_buffer_span`.
     * - De-quantizes and performs the actual reduce operation (op) into `rx_span`.
     *
     * The final result is that `rx_span` accumulates the newly arrived data.
     */
    void runReduceStage(
            ccoip::CCoIPClientState &client_state,
            const uint64_t tag,
            const std::span<const std::byte> &tx_span,
            const std::span<std::byte> &rx_span,
            const std::span<std::byte> &recv_buffer_span,

            const ccoip::ccoip_data_type_t data_type,
            const ccoip::ccoip_data_type_t quantized_type,
            const ccoip::ccoip_quantization_algorithm_t quantization_algorithm,
            const ccoip::ccoip_reduce_op_t op,

            const size_t rank,
            const size_t world_size,
            const std::vector<ccoip_uuid_t> &ring_order,

            const std::optional<ccoip::internal::quantize::DeQuantizationMetaData> &meta_data_self,

            const std::unordered_map<ccoip_uuid_t,
                                     std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_tx_sockets,
            const std::unordered_map<ccoip_uuid_t,
                                     std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_rx_sockets) {
        using namespace tinysockets;
        using namespace ccoip::internal::reduce;
        using namespace ccoip::internal::quantize;

        assert(tx_span.size_bytes() == recv_buffer_span.size_bytes());
        assert(rx_span.size_bytes() % ccoip_data_type_size(data_type) == 0);

        // In a ring, next rank is (rank + 1) % world_size, previous rank is (rank - 1 + world_size) % world_size.
        const size_t rx_peer_idx = (rank + world_size - 1) % world_size;
        const size_t tx_peer_idx = (rank + 1) % world_size;

        const ccoip_uuid_t rx_peer = ring_order.at(rx_peer_idx);
        const ccoip_uuid_t tx_peer = ring_order.at(tx_peer_idx);

        const auto &tx_socket = peer_tx_sockets.at(tx_peer);
        const auto &rx_socket = peer_rx_sockets.at(rx_peer);

        // 1) Send our dequantization meta to the next peer
        {
            ccoip::P2PPacketDequantizationMeta packet{};
            packet.tag = tag;
            packet.dequantization_meta = (meta_data_self
                                              ? *meta_data_self
                                              : DeQuantizationMetaData{});
            if (!tx_socket->sendPacket<ccoip::P2PPacketDequantizationMeta>(packet)) {
                LOG(WARN) << "Failed to send de-quantization meta data!";
                return;
            }
            constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
            client_state.trackCollectiveComsTxBytes(tag, ltv_header + packet.serializedSize());
        }

        // 2) Receive the meta from our previous peer
        DeQuantizationMetaData received_meta_data{}; {
            const auto packet = rx_socket->receivePacket<ccoip::P2PPacketDequantizationMeta>();
            if (!packet) {
                LOG(WARN) << "Failed to receive de-quantization meta data!";
                return;
            }
            received_meta_data = packet->dequantization_meta;
            constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
            client_state.trackCollectiveComsRxBytes(tag, ltv_header + packet->serializedSize());
        }

        // 3) Full-duplex send/recv loop
        size_t bytes_sent = 0;
        size_t bytes_recvd = 0;
        const size_t total_size = tx_span.size_bytes(); // same as recv_buffer_span.size_bytes()

        while (bytes_sent < total_size || bytes_recvd < total_size) {
            // Prepare poll descriptors
            std::vector<poll::PollDescriptor> descriptors;
            descriptors.reserve(2);

            std::optional<poll::PollDescriptor *> tx_desc = std::nullopt;
            std::optional<poll::PollDescriptor *> rx_desc = std::nullopt;

            if (bytes_sent < total_size) {
                descriptors.push_back({*tx_socket, poll::PollEvent::POLL_OUTPUT});
                tx_desc = &descriptors.back();
            }
            if (bytes_recvd < total_size) {
                descriptors.push_back({*rx_socket, poll::PollEvent::POLL_INPUT});
                rx_desc = &descriptors.back();
            }

            if (poll::poll(descriptors, -1) < 0) {
                const std::string error_message = strerror(errno);
                LOG(WARN) << "poll() failed: " << error_message;
                return;
            }

            // 3a) Send if ready
            if (tx_desc && (*tx_desc)->hasEvent(poll::PollEvent::POLL_OUTPUT)) {
                auto send_sub = tx_span.subspan(bytes_sent);
                if (auto sent = send_nonblocking(send_sub, **tx_desc)) {
                    bytes_sent += *sent;
                    client_state.trackCollectiveComsTxBytes(tag, *sent);
                }
            }

            // 3b) Receive if ready
            if (rx_desc && (*rx_desc)->hasEvent(poll::PollEvent::POLL_INPUT)) {
                auto recv_sub = recv_buffer_span.subspan(bytes_recvd);
                if (auto recvd = recv_nonblocking(recv_sub, **rx_desc)) {
                    if (*recvd > 0) {
                        client_state.trackCollectiveComsRxBytes(tag, *recvd);

                        const size_t quant_el_sz = ccoip_data_type_size(quantized_type);

                        // old floor
                        const size_t old_floor = (bytes_recvd / quant_el_sz) * quant_el_sz;
                        bytes_recvd += *recvd;

                        // If new_floor > old_floor, that means we have at least one fully-received element
                        if (const size_t new_floor = (bytes_recvd / quant_el_sz) * quant_el_sz; new_floor > old_floor) {
                            const size_t chunk_bytes = new_floor - old_floor;
                            auto reduce_src_span =
                                    recv_buffer_span.subspan(old_floor, chunk_bytes);
                            // map that to data_type size for the output
                            auto reduce_dst_span = rx_span.subspan(
                                    (old_floor / quant_el_sz) * ccoip_data_type_size(data_type),
                                    (chunk_bytes / quant_el_sz) * ccoip_data_type_size(data_type)
                                    );

                            // We actually "accumulate" it into the rx_span
                            performReduction(reduce_dst_span, reduce_src_span,
                                             data_type, quantized_type,
                                             quantization_algorithm, op,
                                             received_meta_data);
                        }
                    }
                }
            }
        }
    }

    /**
     * \brief Run the "allgather" stage of the ring in a single pipeline step.
     *
     * - Similar to runReduceStage, but there is NO reduce op.  We simply receive
     *   the bytes and copy them into the final buffer.  (Think ring broadcast.)
     */
    void runAllgatherStage(
            ccoip::CCoIPClientState &client_state,
            const uint64_t tag,
            const std::span<const std::byte> &tx_span, // what we send out
            const std::span<std::byte> &rx_span, // where we store newly arrived chunk
            const std::span<std::byte> &recv_buffer_span,

            // In all-gather, the "incoming type" and "local type" are the same.
            const ccoip::ccoip_data_type_t data_type,
            const ccoip::ccoip_data_type_t quantized_type,
            const ccoip::ccoip_quantization_algorithm_t quantization_algorithm,

            const size_t rank,
            const size_t world_size,
            const std::vector<ccoip_uuid_t> &ring_order,

            const std::optional<ccoip::internal::quantize::DeQuantizationMetaData> &meta_data_self,

            const std::unordered_map<ccoip_uuid_t,
                                     std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_tx_sockets,
            const std::unordered_map<ccoip_uuid_t,
                                     std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_rx_sockets) {
        using namespace tinysockets;
        using namespace ccoip::internal::quantize;

        // The only difference from runReduceStage is that we DO NOT do a "performReduction".
        // Instead, we effectively do "performCopy" from the newly arrived data into rx_span.

        const size_t rx_peer_idx = (rank + world_size - 1) % world_size;
        const size_t tx_peer_idx = (rank + 1) % world_size;

        const ccoip_uuid_t rx_peer = ring_order.at(rx_peer_idx);
        const ccoip_uuid_t tx_peer = ring_order.at(tx_peer_idx);

        const auto &tx_socket = peer_tx_sockets.at(tx_peer);
        const auto &rx_socket = peer_rx_sockets.at(rx_peer);

        // 1) Exchange metadata (if you want to keep it consistent, we can do the same handshake)
        {
            ccoip::P2PPacketDequantizationMeta packet{};
            packet.tag = tag;
            packet.dequantization_meta = (meta_data_self
                                              ? *meta_data_self
                                              : DeQuantizationMetaData{});
            if (!tx_socket->sendPacket<ccoip::P2PPacketDequantizationMeta>(packet)) {
                LOG(WARN) << "Failed to send de-quantization meta data in allgather!";
                return;
            }
            size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
            client_state.trackCollectiveComsTxBytes(tag, ltv_header + packet.serializedSize());
        }

        DeQuantizationMetaData received_meta_data{}; {
            const auto packet = rx_socket->receivePacket<ccoip::P2PPacketDequantizationMeta>();
            if (!packet) {
                LOG(WARN) << "Failed to receive de-quant meta in allgather!";
                return;
            }
            received_meta_data = packet->dequantization_meta;
            constexpr size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
            client_state.trackCollectiveComsRxBytes(tag, ltv_header + packet->serializedSize());
        }

        // 2) Full-duplex send/recv
        size_t bytes_sent = 0;
        size_t bytes_recvd = 0;
        const size_t total_size = tx_span.size_bytes(); // same as recv_buffer_span.size_bytes()

        while (bytes_sent < total_size || bytes_recvd < total_size) {
            std::vector<poll::PollDescriptor> descriptors;
            descriptors.reserve(2);

            std::optional<poll::PollDescriptor *> tx_desc = std::nullopt;
            std::optional<poll::PollDescriptor *> rx_desc = std::nullopt;

            if (bytes_sent < total_size) {
                descriptors.push_back({*tx_socket, poll::PollEvent::POLL_OUTPUT});
                tx_desc = &descriptors.back();
            }
            if (bytes_recvd < total_size) {
                descriptors.push_back({*rx_socket, poll::PollEvent::POLL_INPUT});
                rx_desc = &descriptors.back();
            }

            if (poll::poll(descriptors, -1) < 0) {
                const std::string error_message = strerror(errno);
                LOG(WARN) << "poll() failed (allgather): " << error_message;
                return;
            }

            // Send
            if (tx_desc && (*tx_desc)->hasEvent(poll::PollEvent::POLL_OUTPUT)) {
                auto send_sub = tx_span.subspan(bytes_sent);
                if (auto sent = send_nonblocking(send_sub, **tx_desc)) {
                    bytes_sent += *sent;
                    client_state.trackCollectiveComsTxBytes(tag, *sent);
                }
            }

            // Receive
            if (rx_desc && (*rx_desc)->hasEvent(poll::PollEvent::POLL_INPUT)) {
                auto recv_sub = recv_buffer_span.subspan(bytes_recvd);
                if (auto recvd = recv_nonblocking(recv_sub, **rx_desc)) {
                    if (*recvd > 0) {
                        client_state.trackCollectiveComsRxBytes(tag, *recvd);

                        const size_t quant_el_sz = ccoip_data_type_size(quantized_type);

                        const size_t old_floor = (bytes_recvd / quant_el_sz) * quant_el_sz;
                        bytes_recvd += *recvd;

                        if (const size_t new_floor = (bytes_recvd / quant_el_sz) * quant_el_sz; new_floor > old_floor) {
                            const size_t chunk_bytes = new_floor - old_floor;
                            // de-quantize + "copy" into rx_span
                            auto copy_src_span = recv_buffer_span.subspan(old_floor, chunk_bytes);
                            auto copy_dst_span = rx_span.subspan(
                                    (old_floor / quant_el_sz) * ccoip_data_type_size(data_type),
                                    (chunk_bytes / quant_el_sz) * ccoip_data_type_size(data_type));

                            // In allgather, we do not accumulate. We just place the data.
                            ccoip::internal::reduce::performReduction(
                                    copy_dst_span,
                                    copy_src_span,
                                    data_type,
                                    quantized_type,
                                    quantization_algorithm,
                                    ccoip::ccoipOpSet,
                                    received_meta_data
                                    );
                        }
                    }
                }
            }
        }
    }

} // end anonymous namespace

//--------------------------------------------------------
// The main pipeline ring allreduce API
//--------------------------------------------------------
void ccoip::reduce::pipelineRingReduce(
        CCoIPClientState &client_state,
        const uint64_t tag,
        std::span<const std::byte> src_buf,
        const std::span<std::byte> &dst_buf,
        const ccoip_data_type_t data_type,
        const ccoip_data_type_t quantized_type,
        const ccoip_reduce_op_t op,
        const ccoip_quantization_algorithm_t quantization_algorithm,
        const size_t rank,
        const size_t world_size,
        const std::vector<ccoip_uuid_t> &ring_order,
        const std::unordered_map<ccoip_uuid_t,
                                 std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_tx_sockets,
        const std::unordered_map<ccoip_uuid_t,
                                 std::unique_ptr<tinysockets::BlockingIOSocket>> &peer_rx_sockets) {
    using namespace ccoip::internal::quantize;

    // If world_size < 2, no actual reduce needed.
    if (world_size < 2) {
        // Just copy.
        if (dst_buf.data() != src_buf.data()) {
            std::memcpy(dst_buf.data(), src_buf.data(), src_buf.size_bytes());
        }
        // Possibly finalize for op=AVG, etc:
        internal::reduce::performReduceFinalization(dst_buf, data_type, world_size, op);
        return;
    }

    // Check for overlap
    std::optional<std::unique_ptr<std::byte[]>> maybe_src_copy; {
        const auto *src_beg = src_buf.data();
        const auto *src_end = src_beg + src_buf.size_bytes();
        const auto *dst_beg = dst_buf.data();
        const auto *dst_end = dst_beg + dst_buf.size_bytes();

        const bool overlap = !(
            (src_end <= dst_beg) || (dst_end <= src_beg)
        );
        if (overlap) {
            maybe_src_copy = std::make_unique<std::byte[]>(src_buf.size_bytes());
            std::memcpy(maybe_src_copy->get(), src_buf.data(), src_buf.size_bytes());
            src_buf = std::span<const std::byte>(maybe_src_copy->get(), src_buf.size_bytes());
        }
    }

    // Start by copying local data into `dst_buf` so we can accumulate in place.
    std::memcpy(dst_buf.data(), src_buf.data(), src_buf.size_bytes());

    // We will treat the array as world_size contiguous chunks of (approximately)
    // equal size.  (For simplicity here, we assume it evenly divides.)
    // In practice, you might want to do a boundary approach like the gold standard.
    size_t data_type_el_size = ccoip_data_type_size(data_type);
    size_t quant_type_el_size = ccoip_data_type_size(quantized_type);

    assert(dst_buf.size_bytes() % data_type_el_size == 0);

    // how many elements total?
    const size_t total_elements = dst_buf.size_bytes() / data_type_el_size;
    // chunked out among ranks
    size_t chunk_size_elements = total_elements / world_size; // ignoring remainder for brevity
    if (chunk_size_elements == 0) {
        if (rank < world_size) {
            chunk_size_elements = 1;
        }
    }
    if (chunk_size_elements == 0) {
        return; // nothing to do
    }
    const size_t chunk_size_bytes = chunk_size_elements * data_type_el_size;
    const size_t chunk_size_bytes_q = chunk_size_elements * quant_type_el_size;

    // Temporary buffer for inbound data each step
    auto recv_buffer = std::make_unique<std::byte[]>(chunk_size_bytes_q);
    std::span recv_buffer_span(recv_buffer.get(), chunk_size_bytes_q);

    //--------------------------------------------------------------------------
    // PHASE 1: Ring Reduce-Scatter
    //--------------------------------------------------------------------------
    // The standard ring reduce-scatter does world_size - 1 steps.
    // On each step, we choose which chunk to send and which chunk to receive (and reduce).
    // After all steps, rank r ends up with the partial sum for chunk r.
    //--------------------------------------------------------------------------
    for (int step = 0; step < static_cast<int>(world_size) - 1; ++step) {
        // The chunk we send = (rank - step) mod world_size
        const size_t tx_chunk_idx =
                (world_size - step - 1 + rank + 1) % world_size; // as in the incomplete code
        // The chunk we receive = (rank - step - 1) mod world_size
        const size_t rx_chunk_idx =
                (world_size - step - 1 + rank) % world_size;

        // Subspans for the chunk
        std::span<std::byte> rx_span = dst_buf.subspan(rx_chunk_idx * chunk_size_bytes,
                                                       chunk_size_bytes);
        // The source chunk for sending
        std::span<const std::byte> tx_unquantized =
                std::span<const std::byte>(dst_buf)
                .subspan(tx_chunk_idx * chunk_size_bytes, chunk_size_bytes);

        // If quantizing, do it
        std::unique_ptr<std::byte[]> quantized_data;
        std::optional<DeQuantizationMetaData> meta_data;
        if (quantized_type != data_type && quantization_algorithm != ccoipQuantizationNone) {
            quantized_data = std::make_unique<std::byte[]>(chunk_size_bytes_q);
            std::span q_span(quantized_data.get(), chunk_size_bytes_q);

            meta_data = performQuantization(
                    q_span, tx_unquantized,
                    quantization_algorithm, quantized_type, data_type
                    );
            // the final data we send is the quantized data
            tx_unquantized = std::span<const std::byte>(q_span.data(), q_span.size());
        }

        runReduceStage(client_state, tag,
                       /* tx_span    */ tx_unquantized,
                       /* rx_span    */ rx_span,
                       /* recv_buffer_span */ recv_buffer_span,
                       data_type, quantized_type, quantization_algorithm, op,
                       rank, world_size, ring_order,
                       meta_data, // local dequant meta
                       peer_tx_sockets, peer_rx_sockets);
    }

    //--------------------------------------------------------------------------
    // PHASE 2: Ring Allgather Pipeline
    //--------------------------------------------------------------------------
    // Now each rank r has the fully-summed chunk r in `dst_buf[r * chunk_size_bytes, ...]`.
    // We pipeline the distribution so that all ranks eventually get all chunks.
    // We run another world_size - 1 steps of "send chunk, receive next chunk, store it."
    //--------------------------------------------------------------------------
    size_t current_chunk_idx = (rank + 1) % world_size;


    for (int step = 0; step < static_cast<int>(world_size) - 1; ++step) {
        // The chunk we send is the one we "currently" hold
        std::span<const std::byte> tx_span =
                dst_buf.subspan(current_chunk_idx * chunk_size_bytes, chunk_size_bytes);

        // In the classic formula: we will receive chunk
        //   inc_idx = (current_chunk_idx - 1 + world_size) mod world_size
        // once it travels from the previous rank. But let's keep it consistent
        // with the simpler allgather approach:
        const size_t inc_idx = (current_chunk_idx + world_size - 1) % world_size;

        std::span<std::byte> rx_span = dst_buf.subspan(inc_idx * chunk_size_bytes,
                                                       chunk_size_bytes);

        // If quantizing, do so
        std::unique_ptr<std::byte[]> quantized_data;
        std::optional<DeQuantizationMetaData> meta_data;
        if (quantized_type != data_type && quantization_algorithm != ccoipQuantizationNone) {
            quantized_data = std::make_unique<std::byte[]>(chunk_size_bytes_q);
            std::span q_span(quantized_data.get(), chunk_size_bytes_q);
            meta_data = performQuantization(
                    q_span, tx_span,
                    quantization_algorithm, quantized_type, data_type
                    );
            tx_span = std::span<const std::byte>(q_span.data(), q_span.size());
        }

        // We do the ring exchange with no arithmetic reduce:
        runAllgatherStage(client_state, tag,
                          tx_span, rx_span, recv_buffer_span,
                          data_type, quantized_type, quantization_algorithm,
                          rank, world_size, ring_order,
                          meta_data,
                          peer_tx_sockets, peer_rx_sockets);

        // 5) Now that we have *received* the chunk inc_idx,
        //    that chunk is the one we "own" going into the next iteration:
        current_chunk_idx = inc_idx;
    }

    //--------------------------------------------------------------------------
    // Optional finalization for ops that require it (e.g. op=AVG, etc.)
    //--------------------------------------------------------------------------
    internal::reduce::performReduceFinalization(dst_buf, data_type, world_size, op);
}
