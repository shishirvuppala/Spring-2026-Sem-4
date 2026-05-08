#include <stdio.h>
#include <unistd.h>

int main() {
    char ch;

    // TODO: complete the following syscalls
    // read from STDIN_FILENO
    // write to STDOUT_FILENO
    while (read() > 0)
        if (write() < 0) {
            perror("Error writing to stdout\n");
            return 1;
        }

    return 0;
}
