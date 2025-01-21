#include "reduce.hpp"

#include <cassert>
#include <ccoip_master_state.hpp>
#include <cstring>
#include <ccoip_packets.hpp>
#include <ccoip_types.hpp>
#include <quantize.hpp>
#include <reduce_kernels.hpp>
#include <tinysockets.hpp>

static void runReduceStage(
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

        const std::unordered_map<ccoip_uuid_t, std::unique_ptr<tinysockets::BlockingIOSocket>> &
        peer_tx_sockets,
        const std::unordered_map<ccoip_uuid_t, std::unique_ptr<tinysockets::BlockingIOSocket>> &
        peer_rx_sockets) {
    // tx span must be a multiple of the quantized_type size
    // rx span must be a multiple of the data_type size
    assert(tx_span.size_bytes() % ccoip_data_type_size(quantized_type) == 0);
    assert(rx_span.size_bytes() % ccoip_data_type_size(data_type) == 0);
    assert(recv_buffer_span.size_bytes() % ccoip_data_type_size(quantized_type) == 0);
    assert(tx_span.size_bytes() == recv_buffer_span.size_bytes());

    using namespace tinysockets;
    using namespace ccoip::internal::reduce;
    using namespace ccoip::internal::quantize;

    // each peer has one tx and rx peer in the ring
    // the rx peer is always the previous peer in the ring (with wrap-around)
    // the tx peer is always the next peer in the ring (with wrap-around)
    const size_t rx_peer_idx = (rank - 1 + world_size) % world_size;
    const size_t tx_peer_idx = (rank + 1) % world_size;

    const ccoip_uuid_t rx_peer = ring_order.at(rx_peer_idx);
    const ccoip_uuid_t tx_peer = ring_order.at(tx_peer_idx);

    const auto &tx_socket = peer_tx_sockets.at(tx_peer);
    const auto &rx_socket = peer_rx_sockets.at(rx_peer);

    // send meta-data to the next peer
    {
        ccoip::P2PPacketDequantizationMeta packet{};
        packet.tag = tag;
        packet.dequantization_meta = meta_data_self ? *meta_data_self : DeQuantizationMetaData{};
        if (!tx_socket->sendPacket<ccoip::P2PPacketDequantizationMeta>(packet)) {
            LOG(WARN) << "Failed to send de-quantization meta data!";
            return;
        }
        size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
        client_state.trackCollectiveComsTxBytes(tag, ltv_header + packet.serializedSize());
    }

    // receive meta data from prev peer
    // TODO: this is not a good way to do it for multiple concurrent all reduces because you will steal other tag's packet...
    DeQuantizationMetaData received_meta_data{}; {
        const auto packet = rx_socket->receivePacket<ccoip::P2PPacketDequantizationMeta>();
        if (!packet) {
            LOG(WARN) << "Failed to receive de-quantization meta data!";
            return;
        }
        received_meta_data = packet->dequantization_meta;
        size_t ltv_header = sizeof(uint64_t) + sizeof(ccoip::packetId_t);
        client_state.trackCollectiveComsRxBytes(tag, ltv_header + packet->serializedSize());
    }


    // TODO: HINT DATA WITH REDUCE OP TAG?

    // perform concurrent send and receive with fused reduce
    {
        size_t bytes_sent = 0;
        size_t bytes_recvd = 0;

        while (bytes_sent < tx_span.size_bytes() || bytes_recvd < recv_buffer_span.size_bytes()) {
            std::vector<poll::PollDescriptor> descriptors{};
            descriptors.reserve(2);

            std::optional<poll::PollDescriptor *> tx_descriptor = std::nullopt;
            std::optional<poll::PollDescriptor *> rx_descriptor = std::nullopt;
            if (bytes_sent < tx_span.size_bytes()) {
                descriptors.push_back({*tx_socket, poll::PollEvent::POLL_OUTPUT});
                tx_descriptor = &descriptors.back();
            }
            if (bytes_recvd < recv_buffer_span.size_bytes()) {
                descriptors.push_back({*rx_socket, poll::PollEvent::POLL_INPUT});
                rx_descriptor = &descriptors.back();
            }

            if (const int poll_result = poll::poll(descriptors, -1); poll_result < 0) {
                const std::string error_message = strerror(errno);
                LOG(WARN) << "poll() failed: " << error_message;
                return;
            }
            if (tx_descriptor && (*tx_descriptor)->hasEvent(poll::PollEvent::POLL_OUTPUT)) {
                const std::span send_span = tx_span.subspan(bytes_sent);
                if (const auto sent = send_nonblocking(send_span, **tx_descriptor); sent) {
                    bytes_sent += *sent;
                    client_state.trackCollectiveComsTxBytes(tag, *sent);
                }
            }
            if (rx_descriptor && (*rx_descriptor)->hasEvent(poll::PollEvent::POLL_INPUT)) {
                const std::span receive_span = recv_buffer_span.subspan(bytes_recvd);
                if (const auto received = recv_nonblocking(receive_span, **rx_descriptor); received) {
                    if (received > 0) {
                        client_state.trackCollectiveComsRxBytes(tag, *received);
                        const auto r = *received;

                        const size_t quant_element_size = ccoip_data_type_size(quantized_type);

                        // Let "old_bytes_recvd" be how many bytes were read in total so far
                        const size_t old_bytes_recvd_floored =
                                (bytes_recvd / quant_element_size) * quant_element_size;

                        bytes_recvd += r;

                        // Now see how many *complete* elements we have in total
                        const size_t new_bytes_recvd_floored =
                                (bytes_recvd / quant_element_size) * quant_element_size;

                        // The newly arrived, fully aligned portion is [old_bytes_recvd_floored, new_bytes_recvd_floored)
                        if (new_bytes_recvd_floored > old_bytes_recvd_floored) {
                            auto reduce_src_span =
                                    recv_buffer_span.subspan(old_bytes_recvd_floored,
                                                             new_bytes_recvd_floored - old_bytes_recvd_floored);

                            auto reduce_dst_span =
                                    rx_span.subspan( // same mapping to the de-quantized offsets
                                            (old_bytes_recvd_floored / quant_element_size) * ccoip_data_type_size(
                                                    data_type),
                                            (new_bytes_recvd_floored - old_bytes_recvd_floored) / quant_element_size *
                                            ccoip_data_type_size(data_type)
                                            );

                            performReduction(reduce_dst_span, reduce_src_span, data_type, quantized_type,
                                             quantization_algorithm, op, received_meta_data);
                        }
                    }
                }
            }
        }
    }
}

void ccoip::reduce::pipelineRingReduce(
        CCoIPClientState &client_state,
        const uint64_t tag,
        std::span<const std::byte> src_buf, const std::span<std::byte> &dst_buf,
        const ccoip_data_type_t data_type,
        const ccoip_data_type_t quantized_type,
        const ccoip_reduce_op_t op,
        const ccoip_quantization_algorithm_t quantization_algorithm,
        const size_t rank,
        const size_t world_size,
        const std::vector<ccoip_uuid_t> &ring_order,
        const std::unordered_map<ccoip_uuid_t, std::unique_ptr<
                                     tinysockets::BlockingIOSocket>> &
        peer_tx_sockets, const std::unordered_map<ccoip_uuid_t, std::unique_ptr<
                                                      tinysockets::BlockingIOSocket>> &
        peer_rx_sockets) {
    assert(src_buf.size() == dst_buf.size());

    // check if src and dst buffer overlap. if yes, we need a temporary copy of the src buffer
    std::optional<std::unique_ptr<std::byte[]>> src_buf_copy = std::nullopt;
    {
        const void *src_start = src_buf.data();
        const void *src_end = src_buf.data() + src_buf.size_bytes();
        const void *dst_start = dst_buf.data();
        const void *dst_end = dst_buf.data() + dst_buf.size_bytes();

        if ((src_start >= dst_start && src_start < dst_end) ||
            (src_end > dst_start && src_end <= dst_end) ||
            (dst_start >= src_start && dst_start < src_end) ||
            (dst_end > src_start && dst_end <= src_end)) {
            src_buf_copy = std::make_unique<std::byte[]>(src_buf.size_bytes());
            std::memcpy(src_buf_copy->get(), src_buf.data(), src_buf.size_bytes());
            src_buf = std::span(src_buf_copy->get(), src_buf.size_bytes());
        }
    }

    // copy src_buf to dst_buf as initial accumulation
    std::memcpy(dst_buf.data(), src_buf.data(), src_buf.size_bytes());

    const size_t data_type_element_size = ccoip_data_type_size(data_type);
    const size_t quantized_type_element_size = ccoip_data_type_size(quantized_type);

    assert(src_buf.size() % data_type_element_size == 0);
    assert(dst_buf.size() % data_type_element_size == 0);

    const size_t normal_chunk_size_elements = src_buf.size() / world_size / data_type_element_size;
    const size_t normal_chunk_size_bytes = normal_chunk_size_elements * data_type_element_size;
    const size_t normal_chunk_size_bytes_quant = normal_chunk_size_elements * quantized_type_element_size;

    // perform normal reduce stages
    if (world_size > 1) {
        const std::unique_ptr<std::byte[]> recv_buffer(new std::byte[normal_chunk_size_bytes_quant]);
        const std::span recv_buffer_span(recv_buffer.get(), normal_chunk_size_bytes_quant);

        for (int stage = 0; stage < world_size; stage++) {
            // the tx chunk is the chunk we transmit
            // the rx chunk is the chunk we receive
            // they are different subsets of the array.
            // Sending and receiving occurs concurrently to utilize full duplex.
            // The tx & rx chunks is dependent on both rank and stage.
            // They are constructed such that each peer after world size stages has accumulated all data.
            const size_t rx_chunk = (world_size - stage - 1 + rank) % world_size;
            const size_t tx_chunk = (world_size - stage - 1 + rank + 1) % world_size;

            std::span<const std::byte> tx_span = src_buf.
                    subspan(tx_chunk * normal_chunk_size_bytes, normal_chunk_size_bytes);
            const std::span<std::byte> rx_span = dst_buf.subspan(rx_chunk * normal_chunk_size_bytes,
                                                                 normal_chunk_size_bytes);

            std::unique_ptr<std::byte[]> quantized_data = nullptr;
            std::optional<internal::quantize::DeQuantizationMetaData> meta_data = std::nullopt;

            if (quantized_type != data_type && quantization_algorithm != ccoipQuantizationNone) {
                // quantize tx data
                const size_t quantized_data_size = normal_chunk_size_elements * quantized_type_element_size;
                quantized_data = std::make_unique<std::byte[]>(quantized_data_size);

                const std::span quantized_data_span(quantized_data.get(), quantized_data_size);

                meta_data = internal::quantize::performQuantization(quantized_data_span, tx_span,
                                                                    quantization_algorithm,
                                                                    quantized_type,
                                                                    data_type);

                tx_span = quantized_data_span; // the tx span now becomes the quantized data
            }

            runReduceStage(client_state, tag, tx_span, rx_span, recv_buffer_span,
                           data_type, quantized_type, quantization_algorithm,
                           op,
                           rank, world_size,
                           ring_order,
                           meta_data,
                           peer_tx_sockets, peer_rx_sockets);

            // quantized data is automatically deallocated here
        }
    }
    // TODO: handle trailing chunk

    // perform reduce finalization for ops that require it, such as e.g. avg
    internal::reduce::performReduceFinalization(dst_buf, data_type, world_size, op);
}
