#pragma once

#include <string>
#include <span>
#include <vector>
#include <ccoip_types.hpp>

struct ccoip_shared_state_entry_t {
    /// Key of the shared state entry
    std::string key;

    /// Data type of the shared state entry
    ccoip_data_type_t data_type;

    /// References memory for the shared state content.
    /// This memory is owned by the user of the library and will be read or written to
    /// during shared state synchronization depending on whether the peer is distributing
    /// or requesting shared state.
    std::span<std::byte> value;

    /// Whether to verify the identity of the shared state entry.
    /// If true, shared state can differ between peers for this entry.
    bool allow_content_inequality = false;
};

struct ccoip_shared_state_t {
    uint64_t revision;
    std::vector<ccoip_shared_state_entry_t> entries;
};

/// Contains information about a performed shared state synchronization operation
/// such as the number of bytes received
struct ccoip_shared_state_sync_info_t {

    /// Number of bytes received during shared state synchronization
    /// If this value is > 0, that means this peer's shared state was outdated
    /// and thus needed to be updated.
    size_t rx_bytes;

    /// Number of bytes sent during shared state synchronization
    /// If this value is > 0, that means this peer acted as a shared state distributor
    /// for another peer whose shared state was outdated.
    size_t tx_bytes;
};