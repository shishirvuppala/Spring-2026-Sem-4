#include <stdio.h>
#include <unistd.h>

int main() {
    // TODO: create new process using fork and complete if else blocks
    int pid = fork();
    if (pid < 0) {
        printf("fork failed\n");
        return 1;
    } else if (pid == 0) {  // Child
        // TODO: print pid and ppid
        printf("Child :: PID: %d | PPID: %d\n", getpid(), getppid());

    } else {  // Parent
        // TODO: print pid and child pid
        printf("Parent :: PID: %d | Child PID: %d\n", getpid(), pid);
    }

    return 0;
}
