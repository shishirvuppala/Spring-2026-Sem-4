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

    // TODO: open (copy from p2c.c)
    ;

    check_fd(input_fd);

    // TODO: replace stdin with input_fd
    ;

    // TODO: open (copy from p2c.c)
    ;

    check_fd(output_fd);

    // TODO: replace stdout with output_fd
    ;

    // TODO: read, write (copy from p2a.c)
    while (read() > 0) {
        if (write() < 0) {
            perror("Error writing to stdout\n");
            return 1;
        }
    }

    // TODO: close (copy from p2c.c)
    ;
    ;

    return 0;
}
