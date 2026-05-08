#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void check_fd(int fd) {
    if (fd < 0) {
        perror("File descriptor error");
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: ./%s <input_file>\n", argv[0]);
        return 1;
    }

    int input_fd = -1;
    char ch;

    // TODO: open the argv[1] file into input_fd for reading
    ;

    check_fd(input_fd);

    // TODO: complete the following syscalls
    // read from input_fd
    // write to STDOUT_FILENO
    while (read() > 0) {
        if (write() < 0) {
            perror("Error writing to stdout\n");
            return 1;
        }
    }

    // TODO: close the input file descriptor
    ;

    return 0;
}
