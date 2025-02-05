#pragma once

#include <ccoip_client.hpp>
#include <pccl_log.hpp>
#include <memory>
#include <iostream>

#define PCCL_DEBUG(msg) LOG(DEBUG) << msg;

#define __PCCL_STRINGIFY(x) #x
#define __PCCL_TOSTRING(x) __PCCL_STRINGIFY(x)

#define PCCL_VALIDATE(condition, err) {  \
if (!(condition)) {                      \
PCCL_DEBUG(__FILE__ ":" __PCCL_TOSTRING(__LINE__) ": " #condition);         \
return err;                              \
}                                        \
}

#define PCCL_ERR_PROPAGATE(status) { pcclResult_t status_val = status; if (status_val != pcclSuccess) { return (status_val); } }


struct pcclRankInfo_t {
};

struct pcclComm_t {
    std::unique_ptr<ccoip::CCoIPClient> ccoip_client;

    pcclCommCreateParams_t params;

    explicit pcclComm_t(const pcclCommCreateParams_t &params) : params(params) {
    }
};
