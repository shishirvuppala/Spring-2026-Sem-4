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
    input_fd = open(argv[1], O_RDONLY);

    // TODO: open output fd
    output_fd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);

    // TODO: fork a child process
    int pid = fork();
    if (pid < 0) {
        perror("fork failed");
        return 1;
    } else if (pid == 0) {
        // TODO: replace stdin with input_fd
        dup2(input_fd, STDIN_FILENO);

        // TODO: replace stdout with output_fd
        dup2(output_fd, STDOUT_FILENO);

        // TODO: exec cat
        char* args[] = {"cat", NULL};
        execvp(args[0], args);

    } else
        wait(NULL);

    return 0;
}
