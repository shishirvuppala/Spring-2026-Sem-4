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
    input_fd = open(argv[1], O_RDONLY);

    check_fd(input_fd);

    // TODO: complete the following syscalls
    // read from input_fd
    // write to STDOUT_FILENO
    while (read(input_fd, &ch, 1) > 0) {
        if (write(STDOUT_FILENO, &ch, 1) < 0) {
            perror("Error writing to stdout\n");
            return 1;
        }
    }

    // TODO: close the input file descriptor
    close(input_fd);

    return 0;
}
