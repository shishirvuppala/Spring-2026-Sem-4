#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <string.h>

void handler(int sig, siginfo_t *info, void *context) {
    // Get the clock ticks per second for conversion
    long ticks = sysconf(_SC_CLK_TCK);

    // Convert ticks to seconds (double)
    double utime = (double)info->si_utime / ticks;
    double stime = (double)info->si_stime / ticks;

    printf("\nSystem time: %f, User time: %f\n", stime, utime);
    printf("Exit code: %d\n", info->si_status);
    exit(0);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <command> [args...]\n", argv[0]);
        return 1;
    }

    // Set up the signal handler with SA_SIGINFO to get child details
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGCHLD, &sa, NULL);

    pid_t pid = fork();

    if (pid < 0) {
        perror("fork");
        return 1;
    }

    if (pid == 0) {
        // Child process: execute the command
        // argv + 1 skips "./p2c" and passes the rest to execvp
        execvp(argv[1], &argv[1]);
        perror("execvp"); // Only runs if exec fails
        exit(1);
    } else {
        // Parent process: print the command and wait for the signal
        printf("Command: ");
        for (int i = 1; i < argc; i++) printf("%s ", argv[i]);

        // Loop forever; the handler will call exit()
        while (1) pause();
    }

    return 0;
}
