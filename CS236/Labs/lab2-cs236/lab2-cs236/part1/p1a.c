#include <stdio.h>
#include <unistd.h>

int main() {
    int pid = 0;   // TODO: insert syscall here
    int ppid = 0;  // TODO: insert syscall here

    printf("Process ID (PID): %d\n", pid);
    printf("Parent Process ID (PPID): %d\n", ppid);

    printf("Press Enter to continue...\n");
    getchar();

    printf("Running infinite loop. Press Ctrl+C to exit...\n");
    // TODO: insert infinite loop here

    return 0;
}
