#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
    // TODO: fork (copy from p3c.c)
    int pid = fork();
    if (pid < 0) {
        printf("fork failed\n");
        return 1;
    } else if (pid == 0) {  // Child
        // TODO: print child spawned
        printf("Child: %d | Spawned\n", getpid());

        // TODO: sleep (copy from p3c.c)
        sleep(5);

        // TODO: print child exiting
        printf("Child: %d | Exiting\n", getpid());

        // TODO: exit with non-zero status using exit syscall
        exit(42);
    } else {  // Parent
        // TODO: print pid
        printf("Parent: %d | Created child: %d\n", getpid(), pid);

        // TODO: wait for child to terminate and
        // get exit status using waitpid syscall
        int status;
        waitpid(pid, &status, 0);

        // TODO: process and print exit status
        if (WIFEXITED(status)) {
            int exit_status = WEXITSTATUS(status);
            printf("Parent: %d | Reaped child: %d | Exit status: %d\n",
                   getpid(), pid, exit_status);
        } else {
            printf("Parent: %d | Child: %d did not terminate normally\n",
                   getpid(), pid);
        }
    }

    return 0;
}