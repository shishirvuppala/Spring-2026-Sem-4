#include "kernel/types.h"
#include "user/user.h"

int
main(int argc, char* argv[])
{
    int N = 4;
    if (argc > 1) N = atoi(argv[1]);

    int pids[N];

    int ret = tfork2(N, pids);
    if (ret == 0) {  // CHILD; waits for its child
        wait(0);

        printf("child: pid=%d ppid=%d\n", getpid(), getppid());

        exit(0);
    }

    // PARENT; waits for child
    wait(0);

    printf("parent: pid=%d\n", getpid());

    printf("Spawned Chain Summary:\n");
    for (int i = 0; i < N; i++)
        printf("  level %d -> pid %d\n", i + 1, pids[i]);

    exit(0);
}
