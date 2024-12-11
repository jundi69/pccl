#pragma once

#include <ccoip_client.hpp>


struct pcclRankInfo_t {
};

struct pcclComm_t {
    std::unique_ptr<ccoip::CCoIPClient> ccoip_client;
};
