#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

int main() {
    char buffer[BUFFER_SIZE];

    while (1) {
        printf("$ ");
        fflush(stdout);

        // TODO: read input from stdin into buffer
        long bytes_read = -1;
        if (bytes_read <= 0) break;

        buffer[bytes_read - 1] = '\0';
        char* args[] = {buffer, NULL};

        // TODO: fork to create a child process
        int pid = -1;
        if (pid < 0) {
            perror("fork failed");
            continue;
        }

        if (pid == 0) {
            // TODO: in the child process, execute the command in buffer
            int res = -1;
            if (res == -1) {
                perror("exec failed");
                exit(1);
            }
        } else
            wait(NULL);
    }

    printf("\nExiting shell...\n");
    return 0;
}
