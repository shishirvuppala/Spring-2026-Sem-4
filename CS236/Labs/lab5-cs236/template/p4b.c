#include "kernel/types.h"
#include "user/user.h"
#include "kernel/memlayout.h"

#define PGSIZE 4096  // bytes per page
#define MAXVA (1L << (9 + 9 + 9 + 12 - 1))

int main(int argc, char* argv[]) 
{
    uint64 kernbase = KERNBASE;
    uint64 maxva = MAXVA;

    printf("VA TO PA for 10 pages starting from KERNBASE:\n");
    printf("Kernel virtual address => physical address\n");
    for (uint64 addr = kernbase; addr < kernbase + 10 * PGSIZE; addr += PGSIZE){
        uint64 pa = kva_to_pa(addr);
        if (pa == -1) {
            printf("%p => Invalid address\n", (void*)addr);
        } else
            printf("%p => %p\n", (void*)addr, (void*)pa);
    }
    printf("\n\n");

    printf("VA TO PA for 10 pages from MAXVA\n");
    printf("Kernel virtual address => physical address\n");
    for (uint64 addr = maxva - PGSIZE; addr >= maxva - 10 * PGSIZE; addr -= PGSIZE){
        uint64 pa = kva_to_pa(addr);
        if (pa == -1) {
            printf("%p => Invalid address\n", (void*)addr);
        } else
            printf("%p => %p\n", (void*)addr, (void*)pa);
    }
    exit(0);
}
