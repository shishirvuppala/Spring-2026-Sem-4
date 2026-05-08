#include <signal.h>
#include <stdio.h>
#include <unistd.h>

void sigint_handler(int sig) {
    printf("You can't interrupt me :), PID: %d\n", getpid());
}

int main(int argc, char *argv[]) {
    struct sigaction sa;

    /* Set up the signal handler */
    signal(SIGINT,sigint_handler);
    /* Loop indefinitely */
    while (1) {
        pause();  /* Wait for signals */
    }

    return 0;
}
