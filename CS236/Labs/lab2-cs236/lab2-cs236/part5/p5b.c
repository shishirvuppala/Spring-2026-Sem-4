#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: ./%s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    int input_fd, output_fd;

    // TODO: open input fd
    ;

    // TODO: open output fd
    ;

    // TODO: fork a child process
    ;
    if () {  // fork failed
        perror("fork failed");
        return 1;
    } else if () {  // Child
        // TODO: replace stdin with input_fd
        ;

        // TODO: replace stdout with output_fd
        ;

        // TODO: exec cat
        ;
        ;
    } else {  // Parent
        // TODO: wait for child to finish
        ;
    }

    return 0;
}
