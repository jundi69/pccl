# Error Handling & Logging

### Error Returns

Most functions return a `pcclResult_t`, an enum with values like:

- `pcclSuccess` (0)
- `pcclNotInitialized`, `pcclInvalidArgument`, `pcclInvalidUsage`
- `pcclMasterConnectionFailed`, `pcclRankConnectionFailed`, `pcclRankConnectionLost`
- `pcclRemoteError`, etc.

If functions do not explicitly mention a recommended action, it is typical wise to panic or otherwise handle the error
in a non-recoverable way.

## Logging

PCCL uses an internal logging mechanism that writes debug/info/error messages to stdout/stderr, controlled by the
environment variable `PCCL_LOG_LEVEL`. For example:

- `export PCCL_LOG_LEVEL=DEBUG` (Linux/macOS)
- `set PCCL_LOG_LEVEL=DEBUG` (Windows cmd)

Logging can be helpful for diagnosing deadlocks or mismatched states.