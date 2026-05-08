#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int count = 1, N;
pid_t child_pid;

void handler_parent(int sig) {
    if (count > N) return;
    printf("[pid %d]: Parent Received Signal %d\n", getpid(), count++);
    kill(child_pid, SIGUSR1);
}

void handler_child(int sig) {
    if (count > N) return;
    printf("[pid %d]: Child Received Signal %d\n", getpid(), count++);
    kill(getppid(), SIGUSR1);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);

    child_pid = fork();

    if (child_pid > 0) {
        signal(SIGUSR1, handler_parent);
        printf("[pid %d]: Parent Process\n", getpid());
        while (count <= N) pause();
        wait(NULL);
    } else {
        signal(SIGUSR1, handler_child);
        printf("[pid %d]: Child Process\n", getpid());
        while (count <= N) pause();
    }

    return 0;
}
