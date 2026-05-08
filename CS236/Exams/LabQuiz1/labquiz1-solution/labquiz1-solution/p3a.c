#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <path_to_dir>\n", argv[0]);
        return 1;
    }

    if (fork() == 0) {
        execlp("ls", "ls", argv[1], NULL);
        perror("execlp ls");
        exit(1);
    }

    wait(NULL);

    return 0;
}
