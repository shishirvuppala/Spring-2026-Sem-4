#include "kernel/types.h"
#include "user/user.h"
#include "kernel/vm.h"

#define PGSIZE 4096

int main(int argc, char *argv[]) 
{
    printf("Page faults Before sbrk: %ld\n", getpf());
    uint64 va_size = getvasize();
    printf("Virtual address space size before sbrk lazy: %ld bytes\n\n", va_size);

    char *p = sys_sbrk(PGSIZE * 5 , SBRK_LAZY);
    printf("Allocating 5 pages lazily where each page is of size %d bytes...\n", PGSIZE);
    va_size = getvasize();
    printf("Virtual address space size after sbrk lazy: %ld bytes\n\n", va_size);

    for(int i = 0; i < PGSIZE * 5; i += PGSIZE){
        p[i] = 1;
        printf("Touched page at virtual address: %p\n", (void *)(p + i));
        printf("Page faults after touching page %d: %ld\n", i/PGSIZE, getpf());
        printf("PA size after each page touch: %d bytes\n\n", getpasize());
    }

    printf("Total Page faults After touching pages: %ld\n", getpf());
    exit(0);
}
