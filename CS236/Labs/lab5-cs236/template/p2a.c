#include "kernel/types.h"
#include "user/user.h"

/*
 * Lab: Where are my pagetables?
 *
 * This program invokes the system call print_pgdirs(),
 * which prints:
 *   1. The physical address of the Kernel root page table
 *   2. The physical address of the User root page table
 *      of the current process
 *   3. The current value of the satp register
 *
 */

int
main(int argc, char *argv[])
{

    // Invoke system call
    print_pgdirs();

    exit(0);
}