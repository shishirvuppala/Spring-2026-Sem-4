#include "kernel/types.h"
#include "user/user.h"

int
main(int argc, char* argv[])
{
    int N = 4, i;
    if (argc > 1) N = atoi(argv[1]);

    int pids[N];

    int ret = tfork(N, pids);
    if (ret == 0) {  // CHILD
        int child_id = getpid();
        int parent_id = getppid();

        pause(child_id % 10);

        printf("child: pid=%d ppid=%d\n", child_id, parent_id);

        exit(0);
    }

    // ORIGINAL PARENT
    for (i = 0; i < N; i++) wait(0);

    printf("parent: pid=%d\n", getpid());

    printf("Parent Process Summary:\n");
    for (i = 0; i < N; i++) printf("  level %d -> pid %d\n", 1, pids[i]);

    exit(0);
}
