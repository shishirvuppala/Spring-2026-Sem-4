#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

void sigchld_handler(int sig) {
    // TEACHING NOTE: Using 'if' instead of 'while' is a common bug.
    // If two children die at once, this handler only runs once.
    pid_t pid = wait(NULL);
    if (pid > 0) {
        printf("\n[Handler] Reaped child PID: %d\n", pid);
    }
}

int main() {
    signal(SIGCHLD, sigchld_handler);

    for (int i = 0; i < 2; i++) {
        if (fork() == 0) {
            printf("Child %d (PID: %d) started.\n", i + 1, getpid());
            sleep(2); // Both children sleep for the same amount of time
            exit(0);
        }
    }

    printf("Parent (PID: %d) sleeping for 5s, then looping...\n", getpid());
    sleep(5);

    while (1) {
        printf("Parent is idling... press Ctrl+C to stop.\n");
        sleep(2);
    }
    return 0;
}
