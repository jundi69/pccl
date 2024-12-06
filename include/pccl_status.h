#pragma once

// ReSharper disable CppUnusedIncludeDirective
#ifdef __cplusplus
#include <iostream>
#else
#include <stdio.h>
#endif

typedef enum {
    pcclSuccess = 0,
    pcclNotInitialized = 1,
    pcclSystemError = 2,
    pcclInternalError = 3,
    pcclInvalidArgument = 4,
    pcclInvalidUsage = 5,
    pcclRemoteError = 6,
    pcclInProgress = 7,
    pcclNumResults = 8,
    pcclMasterConnectionFailed = 9,
    pcclRankConnectionFailed = 10,
    pcclRankConnectionLost = 11,
    pcclNoSharedStateAvailable = 12,
} pcclResult_t;


#ifdef __cplusplus
#define PCCL_DEBUG(msg) std::cerr << "[OpenNCCL Debug]: " << msg << std::endl
#else
#define PCCL_DEBUG(msg) printf("[OpenNCCL Debug]: %s\n", msg)
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
