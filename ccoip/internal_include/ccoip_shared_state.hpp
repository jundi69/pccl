#pragma once

#include <string>
#include <span>
#include <vector>

struct ccoip_shared_state_entry_t {
    /// Key of the shared state entry
    std::string key;

    /// References memory for the shared state content.
    /// This memory is owned by the user of the library and will be read or written to
    /// during shared state synchronization depending on whether the peer is distributing
    /// or requesting shared state.
    std::span<uint8_t> value;
};

struct ccoip_shared_state_t {
    uint64_t revision;
    std::vector<ccoip_shared_state_entry_t> entries;
};
