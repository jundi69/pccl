#include "ccoip_packet_buffer.hpp"

#include <cstring>
#include <string>

PacketReadBuffer::PacketReadBuffer(const uint8_t *data, const size_t length): data_(data), length_(length),
                                                                              read_index_(0) {
}

std::string PacketReadBuffer::readString() {
    const size_t length = read<uint64_t>();
    if (length > length_ - read_index_) {
        throw std::out_of_range("Read exceeds buffer length");
    }
    std::string result(reinterpret_cast<const char *>(data_ + read_index_), length);
    read_index_ += length;
    return result;
}

void PacketReadBuffer::reset() {
    read_index_ = 0;
}

size_t PacketReadBuffer::remaining() const {
    return length_ - read_index_;
}

void PacketReadBuffer::freeMemory() const {
    delete[] data_;
}

PacketReadBuffer PacketReadBuffer::copy(const PacketReadBuffer &packet_buffer) {
    auto data = new uint8_t[packet_buffer.length_];
    std::memcpy(data, packet_buffer.data_, packet_buffer.length_);
    return {data, packet_buffer.length_};
}

PacketReadBuffer PacketReadBuffer::wrap(uint8_t *data, const size_t length) {
    return {data, length};
}

PacketReadBuffer PacketReadBuffer::wrap(const std::span<uint8_t> &data) {
    return {data.data(), data.size()};
}

// PacketWriteBuffer

PacketWriteBuffer::PacketWriteBuffer(const size_t initial_capacity) {
    if (initial_capacity > 0) {
        data_.reserve(initial_capacity);
    }
}

void PacketWriteBuffer::writeContents(const uint8_t *data, const size_t length) {
    if (data == nullptr) {
        return;
    }
    if (length > SIZE_MAX - data_.size()) {
        throw std::overflow_error("Size overflow in writeContents");
    }
    const size_t old_size = data_.size();
    data_.resize(data_.size() + length);
    std::memcpy(data_.data() + old_size, data, length);
}

void PacketWriteBuffer::writeString(const std::string &str) {
    ensureCapacity(8 + str.size());
    write<uint64_t>(str.size());
    data_.insert(data_.end(), str.begin(), str.end());
}

void PacketWriteBuffer::reset() {
    data_.clear();
}

void PacketWriteBuffer::reserve(const size_t new_capacity) {
    data_.reserve(new_capacity);
}

size_t PacketWriteBuffer::size() const {
    return data_.size();
}

size_t PacketWriteBuffer::capacity() const {
    return data_.capacity();
}

bool PacketWriteBuffer::isAtCapacity() const {
    return data_.size() == data_.capacity();
}

const uint8_t *PacketWriteBuffer::data() const {
    return data_.data();
}

void PacketWriteBuffer::ensureCapacity(const size_t additional_size) {
    if (const size_t required_capacity = data_.size() + additional_size; required_capacity > data_.capacity()) {
        if (required_capacity > SIZE_MAX / 2) {
            throw std::overflow_error("Capacity overflow in ensureCapacity");
        }
        data_.reserve(required_capacity * 2); // Safe from overflow
    }
}
