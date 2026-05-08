#include "kernel/types.h"
#include "user/user.h"

int main(int argc, char *argv[])
{
    char *var = (char *)sbrk(8192); // Allocate 8192 bytes (2 pages)

    uint64 user_va1, user_va2, user_va3;
    uint64 user_pa1, user_pa2, user_pa3;
    // Implement the logic for pte accordingly
    // uint64 user_pte1, user_pte2, user_pte3;

    // Virtual address of a user-space variable
    user_va1 = (uint64)var;
    user_va2 = (uint64)(var + 1);
    user_va3 = (uint64)(var + 4096);

    // Translate user virtual address to physical address
    user_pa1 = va_to_pa(user_va1);
    user_pa2 = va_to_pa(user_va2);
    user_pa3 = va_to_pa(user_va3);

    printf("User VA  : %p\n", (void *)user_va1);
    printf("User PA  : %p\n", (void *)user_pa1);
    printf("---\n");
    printf("User VA  : %p\n", (void *)user_va2);
    printf("User PA  : %p\n", (void *)user_pa2);
    printf("---\n");
    printf("User VA  : %p\n", (void *)user_va3);
    printf("User PA  : %p\n", (void *)user_pa3);

    exit(0);
}
