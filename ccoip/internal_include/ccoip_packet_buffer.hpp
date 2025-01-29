#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <span>
#include <bit>
#include <algorithm>

#include <type_utils.hpp>


template<typename T>
concept BufferPOD = std::is_arithmetic_v<T>;

/**
 * @class PacketReadBuffer
 * @brief A utility class for reading data from a byte buffer in big-endian format.
 *
 * PacketBuffer wraps a raw buffer (uint8_t*) with a specified length, providing methods to read various data types.
 */
class PacketReadBuffer final {
public:
    PacketReadBuffer(const uint8_t *data, size_t length);

    /**
     * @brief Reads a value of type T from the buffer in big-endian format.
     *
     * T must be an arithmetic type (e.g., int, float, uint32_t).
     * @tparam T The type to read from the buffer.
     * @return The value of type T read from the buffer.
     * @throws std::out_of_range if reading exceeds the buffer length.
     */
    template<typename T> requires BufferPOD<T>
    T read() {
        if (read_index_ + sizeof(T) > length_) {
            throw std::out_of_range("Read exceeds buffer length");
        }

        T value;
        if constexpr (std::is_integral_v<T>) {
            value = readIntegral<T>();
        } else if constexpr (std::is_same_v<T, float>) {
            value = std::bit_cast<T>(readIntegral<uint32_t>());
        } else if constexpr (std::is_same_v<T, double>) {
            value = std::bit_cast<T>(readIntegral<uint64_t>());
        } else {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Unsupported type");
        }

        read_index_ += sizeof(T);
        return value;
    }
private:
    template<typename T> requires std::is_integral_v<T>
    T readIntegral() {
        T value = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            value = (value << 8) | data_[read_index_ + i];
        }
        return value;
    }
public:

    void readContents(uint8_t *dst, const size_t length) {
        if (read_index_ + length > length_) {
            throw std::out_of_range("Read exceeds buffer length");
        }
        std::copy_n(data_ + read_index_, length, dst);
        read_index_ += length;
    }

    /**
     * @brief Reads a fixed-length array of type T from the buffer.
     *
     * @tparam T The element type of the fixed-length array.
     * @return The array of type T read from the buffer.
     * @throws std::out_of_range if reading exceeds the bounded array length.
     */
    template<typename T, size_t N> requires BufferPOD<T>
    std::array<T, N> readFixedArray() {
        std::array<T, N> array;
        for (size_t i = 0; i < N; ++i) {
            array[i] = read<T>();
        }
        return array;
    }

    /**
     * @brief Reads a variable-length list of type T from the buffer.
     * @tparam T The type of the list to read from the buffer.
     * @return The list of type T read from the buffer.
     */
    template<typename T> requires BufferPOD<T>
    std::vector<T> readVarLenList() {
        const auto list_size = read<uint32_t>();
        std::vector<T> list;
        list.reserve(list_size);
        for (size_t i = 0; i < list_size; ++i) {
            if constexpr (is_std_array_v<T>) {
                list.emplace_back(readFixedArray<typename extract_array_type<T>::type, extract_array_type<T>::size>());
            } else {
                list.emplace_back(read<T>());
            }
        }
        return list;
    }

    /**
     * @brief Reads a string of the specified length from the buffer.
     *
     * reads the number of expected bytes from the buffer and converts it to a string.
     * @throws std::out_of_range if reading exceeds the buffer length.
     */
    std::string readString();

    /**
     * @brief Resets the read position to the beginning of the buffer.
     */
    void reset();

    /**
     * @return The number of bytes remaining in the buffer.
     */
    [[nodiscard]] size_t remaining() const;

    /**
     * Frees the memory allocated for the buffer.
     * This method should only be used for read buffers created via `copy`, which allocates memory.
     * The buffer created via `wrap` does not allocate memory and should not be freed because its intended
     * use is to wrap existing data.
     */
    void freeMemory() const;

    /**
     * Creates a copy of the PacketReadBuffer with the underlying data copied.
     * The user is responsible for freeing the memory of the copied buffer.
     * @param packet_buffer The PacketReadBuffer to copy.
     * @return A new PacketReadBuffer instance with a copy of the data.
     */
    [[nodiscard]] static PacketReadBuffer copy(const PacketReadBuffer &packet_buffer);

    /**
     * @brief Static factory method to wrap an existing byte array in a PacketBuffer.
     *
     * @param data Pointer to the raw byte buffer.
     * @param length The length of the buffer.
     * @return A PacketBuffer instance wrapping the provided data.
     */
    [[nodiscard]] static PacketReadBuffer wrap(uint8_t *data, size_t length);

    /**
      * @brief Static factory method to wrap an existing byte array in a PacketBuffer.
      *
      * @param data Pointer to the raw byte buffer.
      * @return A PacketBuffer instance wrapping the provided data.
      */
    [[nodiscard]] static PacketReadBuffer wrap(const std::span<uint8_t> &data);

private:
    const uint8_t *data_;
    size_t length_;
    size_t read_index_;
};

/**
 * @class PacketWriteBuffer
 * @brief A utility class for dynamically writing data to a byte buffer in big-endian format.
 *
 * PacketWriteBuffer manages a dynamically growing buffer, providing methods to write various data types
 * in big-endian format. The buffer automatically grows as needed and can be preallocated with the `reserve` function.
 */
class PacketWriteBuffer {
public:
    /**
     * @brief Constructs a PacketWriteBuffer with an optional initial capacity.
     * @param initial_capacity The initial capacity of the buffer.
     */
    explicit PacketWriteBuffer(size_t initial_capacity = 0);

    /**
     * @brief Writes a value of type T to the buffer in big-endian format.
     *
     * T must be an arithmetic type (e.g., int, float, uint32_t).
     * @tparam T The type to write to the buffer.
     * @param value The value to write to the buffer.
     */
    template<typename T> requires BufferPOD<T>
    void write(T value) {
        if constexpr (std::is_same_v<T, float>) {
            // if T is "float", bitcast to u32
            writeIntegral<uint32_t>(std::bit_cast<uint32_t>(value));
        } else if constexpr (std::is_same_v<T, double>) {
            // if T is "double", bitcast to u64
            writeIntegral<uint64_t>(std::bit_cast<uint64_t>(value));
        } else {
            writeIntegral<T>(value);
        }
    }

private:
    template<typename T> requires std::is_integral_v<T>
    void writeIntegral(T value) {
        ensureCapacity(sizeof(T));

        for (size_t i = 0; i < sizeof(T); ++i) {
            data_.push_back(static_cast<uint8_t>(value >> (8 * (sizeof(T) - 1 - i))));
        }
    }

public:
    /**
     * @brief Writes a bounded array of type T to the buffer.
     *
     * T must be a bounded array with a fixed size (e.g., std::array<T, 4>, std::array<T, 8>)
     * @tparam T The type of the array to write to the buffer.
     * @param array The array to write to the buffer.
     */
    template<typename T, size_t N> requires BufferPOD<T>
    void writeFixedArray(std::array<T, N> array) {
        for (size_t i = 0; i < N; ++i) {
            write(array[i]);
        }
    }

    /**
     * @brief Writes a variable-length list of type T to the buffer.
     *
     * @tparam T element type of the list.
     * @param list The list to write to the buffer.
     */
    template<typename T>
    void writeVarLenList(const std::vector<T> &list) {
        write<uint32_t>(list.size());
        for (const auto &item: list) {
            // if T is a std::array, write a fixed-length array
            if constexpr (is_std_array_v<T>) {
                writeFixedArray<typename extract_array_type<T>::type, extract_array_type<T>::size>(item);
            } else {
                write(item);
            }
        }
    }

    /**
     * @brief Copies raw data to the buffer.
     */
    void writeContents(const uint8_t *data, size_t length);

    /**
     * @brief Writes a string to the buffer.
     *
     * This method writes each character in the string as a byte.
     * @param str The string to write to the buffer.
     */
    void writeString(const std::string &str);

    /**
     * @brief Resets the write position to the beginning of the buffer.
     *
     * Clears the buffer, allowing it to be reused.
     */
    void reset();

    /**
     * @brief Reserves a minimum capacity for the buffer to avoid frequent reallocations.
     * @param new_capacity The minimum capacity to reserve.
     */
    void reserve(size_t new_capacity);

    /**
     * @brief Returns the number of bytes written to the buffer.
     * @return The size of the buffer.
     */
    [[nodiscard]] size_t size() const;

    /**
     * @brief Returns the number of bytes that are currently allocated in the buffer.
     * @return The capacity of the buffer.
     */
    [[nodiscard]] size_t capacity() const;

    /**
     * @breif Returns whether the buffer is fully filled to its currently allocated capacity.
     * A buffer that is at capacity will require a reallocation to write additional data.
     * @return True if the buffer is at capacity, false otherwise.
     */
    [[nodiscard]] bool isAtCapacity() const;


    /**
     * @brief Accesses the underlying data buffer.
     * @return A pointer to the raw data in the buffer.
     */
    [[nodiscard]] const uint8_t *data() const;

private:
    /**
     * @brief Ensures that the buffer has enough capacity to write additional bytes.
     * @param additional_size The additional number of bytes required.
     */
    void ensureCapacity(size_t additional_size);

    std::vector<uint8_t> data_{}; ///< Underlying dynamic buffer.
};
