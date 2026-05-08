#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <path_to_dir>\n", argv[0]);
        return 1;
    }

    int pipe_fd[2];

    if (pipe(pipe_fd) == -1) {
        perror("pipe");
        exit(1);
    }

    if (fork() == 0) {
        close(pipe_fd[0]);
        dup2(pipe_fd[1], STDOUT_FILENO);
        execlp("ls", "ls", argv[1], NULL);
        perror("execlp ls");
        exit(1);
    }

    if (fork() == 0) {
        close(pipe_fd[1]);
        dup2(pipe_fd[0], STDIN_FILENO);
        execlp("wc", "wc", "-l", NULL);
        perror("execlp wc");
        exit(1);
    }

    close(pipe_fd[0]);
    close(pipe_fd[1]);

    wait(NULL);
    wait(NULL);

    return 0;
}
