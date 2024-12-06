#include <pccl.h>
#include <assert.h>

int main(void) {
    assert(pcclInit() == pcclSuccess);
    return 0;
}
