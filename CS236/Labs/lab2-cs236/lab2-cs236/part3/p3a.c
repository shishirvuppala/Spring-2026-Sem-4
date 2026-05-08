#include <stdio.h>
#include <unistd.h>

int main() {
    fork();
    fork();
    fork();

    printf("Process ID (PID): %d\n", getpid());

    return 0;
}
