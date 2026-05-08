#include "kernel/types.h"
#include "user/user.h"
#include "kernel/vm.h"

int main(int argc, char *argv[])
{
    printf("----------------SBRK LAZY:----------------\n");
    int pasize = getpasize();
    printf("Physical memory size before: %d bytes\n", pasize);
    int vasize = getvasize();
    printf("Virtual memory size before: %d bytes\n\n", vasize);
    printf("Allocating 4096 bytes lazily...\n");
    sys_sbrk(4096, SBRK_LAZY);
    pasize = getpasize();
    printf("Physical memory size after: %d bytes\n", pasize);
    vasize = getvasize();
    printf("Virtual memory size after: %d bytes\n\n", vasize);

    printf("----------------SBRK EAGER:---------------\n");
    pasize = getpasize();
    printf("Physical memory size before: %d bytes\n", pasize);
    vasize = getvasize();
    printf("Virtual memory size before: %d bytes\n\n", vasize);
    printf("Allocating 4096 bytes eagerly...\n");
    sys_sbrk(4096, SBRK_EAGER);
    pasize = getpasize();
    printf("Physical memory size after: %d bytes\n", pasize);
    vasize = getvasize();
    printf("Virtual memory size after: %d bytes\n\n", vasize);

    exit(0);
}
