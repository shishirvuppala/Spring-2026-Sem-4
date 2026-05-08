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
    if (argc != 3) {
        printf("Usage: ./%s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    int input_fd = -1;
    int output_fd = -1;
    char ch;

    // TODO: open (copy from p2b.c)
    ;

    check_fd(input_fd);

    // TODO: open the argv[2] file into output_fd for writing
    // (create it if it doesn't exist, truncate it if it does)
    ;

    check_fd(output_fd);

    // TODO: complete the following syscalls
    // read from input_fd
    // write to output_fd
    while (read() > 0) {
        if (write() < 0) {
            perror("Error writing to stdout\n");
            return 1;
        }
    }

    // TODO: close the input and output file descriptors
    ;
    ;

    return 0;
}
