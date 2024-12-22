#pragma once

#include <memory>

template<typename Derived, typename Base>
Derived *dynamic_unique_ptr_cast(std::unique_ptr<Base> &base) {
    if (auto derived = dynamic_cast<Derived *>(base.get())) {
        return derived;
    }
    return nullptr;
}