#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <pid>\n", argv[0]);
        exit(1);
    }

    int target_pid = atoi(argv[1]);

    if (kill(target_pid, SIGUSR1) == -1) {
        perror("Error sending signal");
        exit(1);
    }

    printf("Signal %d sent to process %d\n", SIGUSR1, target_pid);
    return 0;
}