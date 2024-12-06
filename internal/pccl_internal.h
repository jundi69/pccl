#pragma once

#include <ccoip_client_handler.hpp>


struct pcclRankInfo_t {
};

struct pcclComm_t {
    std::unique_ptr<ccoip::CCoIPClientHandler> ccoip_handler;
};
