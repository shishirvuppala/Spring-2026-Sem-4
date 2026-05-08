#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <path_to_dir>\n", argv[0]);
        return 1;
    }

    // TODO: execute ls command for the given directory in a child process

    // TODO: Parent: Reap the child

    return 0;
}
