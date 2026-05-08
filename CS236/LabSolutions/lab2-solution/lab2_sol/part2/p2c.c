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

    // TODO: (copy from p2b.c)
    input_fd = open(argv[1], O_RDONLY);

    check_fd(input_fd);

    // TODO: open the argv[2] file into output_fd for writing
    // (create it if it doesn't exist, truncate it if it does)
    output_fd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);

    check_fd(output_fd);

    // TODO: complete the following syscalls
    // read from input_fd
    // write to output_fd
    while (read(input_fd, &ch, 1) > 0) {
        if (write(output_fd, &ch, 1) < 0) {
            perror("Error writing to stdout\n");
            return 1;
        }
    }

    // TODO: close the input and output file descriptors
    close(input_fd);
    close(output_fd);

    return 0;
}
