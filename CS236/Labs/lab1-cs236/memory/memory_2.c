#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define ARRAY_SIZE 1 << 21

int main() 
{
    int* array;
    int i;

    printf("\n\nProgram : 'memory_1'\n");
    printf("____________________\n");
    printf("\n\nPID : %d \n", getpid());
    printf("Size of int : %ld \n", sizeof(int));
    printf("\nCheck VmSize and VmRSS.\nPress Enter Key after check.\n");
    getchar();

    array = (int*)malloc(sizeof(int) * ARRAY_SIZE);

    printf("\nCheck VmSize and VmRSS\nPress Enter Key to exit.\n");
    getchar();
    return 0;
}
