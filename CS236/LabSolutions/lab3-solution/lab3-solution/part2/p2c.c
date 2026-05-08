#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

/* Global variables to store timing info from signal handler */
volatile sig_atomic_t child_exited = 0;
int child_exit_code = 0;
double user_time = 0.0;
double sys_time = 0.0;

void sigchld_handler(int sig, siginfo_t *info, void *context) {
    int status;
    struct rusage usage;
    pid_t pid;

    /* Reap the child process and get resource usage */
    pid = wait3(&status, WNOHANG, &usage);
    if (pid > 0) {
        /* Get exit code */
        if (WIFEXITED(status)) {
            child_exit_code = WEXITSTATUS(status);
        } else {
            child_exit_code = -1;
        }

        /* Get user and system time from rusage */
        user_time = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1000000.0;
        sys_time = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1000000.0;

        child_exited = 1;
    }
}

int main(int argc, char *argv[]) {
    pid_t pid;
    struct sigaction sa;
    int i;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <command> [args...]\n", argv[0]);
        exit(1);
    }

    /* Set up the SIGCHLD handler using sigaction with SA_SIGINFO */
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = sigchld_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;  /* Use sa_sigaction instead of sa_handler */

    if (sigaction(SIGCHLD, &sa, NULL) == -1) {
        perror("sigaction");
        exit(1);
    }

    pid = fork();

    if (pid < 0) {
        perror("fork");
        exit(1);
    } else if (pid == 0) {
        /* Child process - exec the command */
        /* argv+1 points to the command and its arguments */
        execvp(argv[1], argv + 1);
        /* If exec fails */
        perror("exec");
        exit(127);
    } else {
        /* Parent process */
        /* Print the command */
        printf("Command: ");
        for (i = 1; i < argc; i++) {
            printf("%s", argv[i]);
            if (i < argc - 1) {
                printf(" ");
            }
        }
        printf("\n");

        /* Wait for the child to exit */
        while (!child_exited) {
            pause();
        }

        /* Print timing information */
        printf("System time: %f, User time: %f\n", sys_time, user_time);
        printf("Exit code: %d\n", child_exit_code);
    }

    return 0;
}
