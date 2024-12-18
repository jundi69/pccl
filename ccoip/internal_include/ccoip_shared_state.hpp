#pragma once

#include <string>
#include <span>
#include <vector>

enum ccoip_data_type_t {
    ccoipUint8 = 0,
    ccoipInt8 = 1,
    ccoipUint16 = 2,
    ccoipUint32 = 3,
    ccoipInt16 = 4,
    ccoipInt32 = 5,
    ccoipUint64 = 6,
    ccoipInt64 = 7,
    ccoipFloat = 8,
    ccoipDouble = 9,
};

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
