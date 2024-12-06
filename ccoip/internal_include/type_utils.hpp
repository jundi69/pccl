#pragma once

#include <array>

template<typename T>
constexpr bool is_std_array_v = false;

template<typename T, std::size_t N>
constexpr bool is_std_array_v<std::array<T, N>> = true;

// Helper struct to extract T and N from std::array<T, N>
template<typename T>
struct extract_array_type;

template<typename T, std::size_t N>
struct extract_array_type<std::array<T, N>> {
    using type = T;
    static constexpr std::size_t size = N;
};
