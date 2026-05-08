#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <path_to_dir>\n", argv[0]);
        return 1;
    }

    // TODO: Parent: Create a pipe for inter-process communication

    // if (___) child-1?
    // {
    //     - TODO: Make sure as a first step you close the end/ends of the
    //       pipe that you will not need for this child process
    // }

    // if (___) child-2?
    // {
    //     - TODO: Make sure as a first step you close the end/ends of the
    //       pipe that you will not need for this child process
    // }

    // TODO: Parent: Close all pipe ends

    // TODO: Parent: Reap both child process

    return 0;
}
