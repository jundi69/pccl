#pragma once

#include <cstdint>
#include <optional>
#include <atomic>

template<typename V>
class LockFreeMap {
    std::atomic_uint64_t *keys;
    std::atomic<V *> *values;
    std::atomic_uint64_t insert_pos{0};
    size_t max_entries;
public:
    explicit LockFreeMap(const size_t max_entries) : keys(new std::atomic_uint64_t[max_entries]{UINT64_MAX}),
                                                     values(new std::atomic<V *>[max_entries]{nullptr}),
                                                     max_entries(max_entries) {
    }

    /// @warning tag unique to thread id
    void emplace(const uint64_t key, V value) {
        if (key == UINT64_MAX) {
            throw std::invalid_argument("key cannot be UINT64_MAX, as it is a reserved value");
        }
        // first check if the key already exists
        {
            const auto n = insert_pos.load(std::memory_order_seq_cst);
            for (size_t i = 0; i < n; i++) {
                if (keys[i].load(std::memory_order_seq_cst) == key) {
                    delete values[i].exchange(new V(std::move(value)), std::memory_order_seq_cst);
                    return;
                }
            }
        }
        {
            const auto pos = insert_pos.fetch_add(1, std::memory_order_seq_cst);
            if (pos >= max_entries) {
                throw std::runtime_error("LockFreeMap is full! Please increase the capacity");
            }
            values[pos].store(new V(std::move(value)), std::memory_order_seq_cst);
            keys[pos].store(key, std::memory_order_seq_cst);
        }
    }

    std::optional<V *> get(const uint64_t key) {
        if (key == UINT64_MAX) {
            throw std::invalid_argument("key cannot be UINT64_MAX, as it is a reserved value");
        }
        const auto n = insert_pos.load(std::memory_order_seq_cst);
        for (size_t i = 0; i < n; i++) {
            if (keys[i].load(std::memory_order_seq_cst) == key) {
                auto ptr = values[i].load(std::memory_order_seq_cst);
                if (ptr != nullptr) {
                    return ptr;
                }
                return std::nullopt;
            }
        }
        return std::nullopt;
    }

    V *getOrCreate(const uint64_t key) {
        if (key == UINT64_MAX) {
            throw std::invalid_argument("key cannot be UINT64_MAX, as it is a reserved value");
        }
        const auto n = insert_pos.load(std::memory_order_seq_cst);
        for (size_t i = 0; i < n; i++) {
            if (keys[i].load(std::memory_order_seq_cst) == key) {
                return values[i].load(std::memory_order_seq_cst);
            }
        }
        const auto pos = insert_pos.fetch_add(1, std::memory_order_seq_cst);
        if (pos >= max_entries) {
            throw std::runtime_error("LockFreeMap is full! Please increase the capacity");
        }
        keys[pos].store(key, std::memory_order_seq_cst);
        values[pos].store(new V(), std::memory_order_seq_cst);
        V *ptr = values[pos].load(std::memory_order_seq_cst);
        return ptr;
    }

    [[nodiscard]] bool contains(const uint64_t key) const {
        if (key == UINT64_MAX) {
            throw std::invalid_argument("key cannot be UINT64_MAX, as it is a reserved value");
        }
        const auto n = insert_pos.load(std::memory_order_seq_cst);
        for (size_t i = 0; i < n; i++) {
            if (keys[i].load(std::memory_order_seq_cst) == key) {
                const auto ptr = values[i].load(std::memory_order_seq_cst);
                if (ptr != nullptr) {
                    return true;
                }
            }
        }
        return false;
    }

    /// @warning erase must not be called concurrently with get accesses for the same key
    /// to avoid dereference of a pointer that was freed if unlucky
    void erase(const uint64_t key) {
        if (key == UINT64_MAX) {
            throw std::invalid_argument("key cannot be UINT64_MAX, as it is a reserved value");
        }
        const auto n = insert_pos.load(std::memory_order_seq_cst);
        for (size_t i = 0; i < n; i++) {
            if (keys[i].load(std::memory_order_seq_cst) == key) {
                delete values[i].exchange(nullptr, std::memory_order_seq_cst);
                return;
            }
        }
    }

    ~LockFreeMap() {
        delete[] keys;
        for (int i = 0; i < max_entries; i++) {
            const auto ptr = values[i].load(std::memory_order_seq_cst);
            delete ptr;
        }
        delete[] values;
    }
};
