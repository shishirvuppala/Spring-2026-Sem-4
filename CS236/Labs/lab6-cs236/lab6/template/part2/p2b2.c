#include "kernel/types.h"
#include "user/user.h"

int
main(void)
{
    int pid = fork();
    if (pid < 0) {
        printf("fork failed\n");
        exit(1);
    }

    if (pid == 0) {
        if (shm_init() < 0) {
            printf("[Child] shm_init failed\n");
            exit(1);
        }
        printf("[Child] shm_init succeeded\n");

        char* p = (char*)shm_attach();
        if (p == 0) {
            printf("[Child] shm_attach failed\n");
            exit(1);
        }
        printf("[Child] shm_attach succeeded\n");

        strcpy(p, "Message from child");
        printf("[Child] wrote: %s\n", p);

        if (shm_detach() < 0) {
            printf("[Child] shm_detach failed\n");
            exit(1);
        }
        printf("[Child] shm_detach succeeded\n");

        exit(0);
    } else {
        wait(0);

        char* p = (char*)shm_attach();
        if (p == 0) {
            printf("[Parent] shm_attach failed\n");
            exit(1);
        }
        printf("[Parent] shm_attach succeeded\n");

        printf("[Parent] reads: %s\n", p);

        if (shm_detach() < 0) {
            printf("[Parent] shm_detach failed\n");
            exit(1);
        }
        printf("[Parent] shm_detach succeeded\n");

        if (shm_destroy() < 0) {
            printf("[Parent] shm_destroy failed\n");
            exit(1);
        }
        printf("[Parent] shm_destroy succeeded\n");

        exit(0);
    }
}
