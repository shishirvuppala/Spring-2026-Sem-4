#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int count = 1;

void handler_parent(int sig) {
    printf("[pid %d]: Parent Received Signal %d\n", getpid(), count++);
}

void handler_child(int sig) {
    printf("[pid %d]: Child Received Signal %d\n", getpid(), count++);
}

int main(int argc, char* argv[]) {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(1);

    }

    if (pid != 0) {
        signal(SIGUSR1, handler_parent);
        printf("[pid %d]: Parent Process\n", getpid());
        while (1) pause();
        wait(NULL);
    } else {
        signal(SIGUSR1, handler_child);
        printf("[pid %d]: Child Process\n", getpid());
        while (1) pause();
    }

    return 0;
}
