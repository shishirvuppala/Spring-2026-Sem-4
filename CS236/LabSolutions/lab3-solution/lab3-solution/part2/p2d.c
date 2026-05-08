#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

void sigint_handler(int sig) {
    printf("Received SIGINT, PID: %d\n", getpid());
}

int main(int argc, char *argv[]) {
    // Set up the signal handler for the parent (and initially the child)
    signal(SIGINT, sigint_handler);

    pid_t pid = fork();

    if (pid < 0) {
        perror("fork failed");
        return 1;
    }

    if (pid == 0) {
        /* CHILD PROCESS */

        // Move the child to a new process group
        // setpgid(0, 0) sets the PGID of the calling process to its own PID
        if (setpgid(0, 0) == -1) {
            perror("setpgid failed");
            return 1;
        }

        printf("Child Process: PID: %d, PGID: %d\n", getpid(), getpgrp());

        while (1) {
            pause();
        }
    } else {
        /* PARENT PROCESS */
        printf("Parent Process: PID: %d, PGID: %d\n", getpid(), getpgrp());
        printf("Child created with PID: %d\n", pid);

        while (1) {
            pause();
        }
    }

    return 0;
}
