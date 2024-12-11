#pragma once

#include <ccoip_client.hpp>

#ifdef __cplusplus
#include <iostream>
#else
#include <stdio.h>
#endif

#ifdef __cplusplus
#define PCCL_DEBUG(msg) std::cerr << "[PCCL Debug]: " << msg << std::endl
#else
#define PCCL_DEBUG(msg) printf("[PCCL Debug]: %s\n", msg)
#endif

#define __PCCL_STRINGIFY(x) #x
#define __PCCL_TOSTRING(x) __PCCL_STRINGIFY(x)

#define PCCL_VALIDATE(condition, err) { \
if (!(condition)) {                      \
PCCL_DEBUG(__FILE__ ":" __PCCL_TOSTRING(__LINE__) ": " #condition);         \
return err;                          \
}                                        \
}

#define PCCL_ERR_PROPAGATE(status) { pcclResult_t status_val = status; if (status_val != pcclSuccess) { return (status_val); } }


struct pcclRankInfo_t {
};

struct pcclComm_t {
    std::unique_ptr<ccoip::CCoIPClient> ccoip_client;
};
