#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
    // TODO: fork and complete if-else (copy from p3b.c)
    int pid = fork();
    if (pid < 0) {
        printf("fork failed\n");
        return 1;
    } else if (pid == 0) {  // Child
        // TODO: print (copy from p3b.c)
        printf("Child :: PID: %d | PPID: %d\n", getpid(), getppid());

        // TODO: sleep for 5 secs using sleep syscall
        sleep(5);

        printf("Child process terminating...\n");
    } else {  // Parent
        // TODO: wait for user input before waiting for child
        getchar();

        printf("Parent process waiting for child to terminate...\n");

        // TODO: wait for child to terminate using wait syscall
        wait(NULL);

        printf("Child process terminated!\n");

        // TODO: print (copy from p3b.c)
        printf("Parent :: PID: %d | Child PID: %d\n", getpid(), pid);
    }

    return 0;
}
