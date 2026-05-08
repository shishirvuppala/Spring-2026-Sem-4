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

    // TODO: (copy from p2c.c)
    input_fd = open(argv[1], O_RDONLY);

    check_fd(input_fd);

    // TODO: replace stdin with input_fd
    dup2(input_fd, STDIN_FILENO);

    // TODO: (copy from p2c.c)
    output_fd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);

    check_fd(output_fd);

    // TODO: replace stdout with output_fd
    dup2(output_fd, STDOUT_FILENO);

    // TODO: (copy from p2a.c)
    while (read(STDIN_FILENO, &ch, 1) > 0) {
        if (write(STDOUT_FILENO, &ch, 1) < 0) {
            perror("Error writing to stdout\n");
            return 1;
        }
    }

    // TODO: close the input and output file descriptors
    close(input_fd);
    close(output_fd);

    return 0;
}
