#include "kernel/types.h"
#include "user/user.h"

int
main(int argc, char* argv[])
{
    int N = 2, i;
    if (argc > 1) N = atoi(argv[1]);

    // Initialize shared memory (refcnt should become 0)
    printf("[Parent] Calling shm_init()...\n");
    if (shm_init() < 0) {
        printf("[Parent] ERROR: shm_init failed\n");
        exit(1);
    }
    printf("[Parent] shm_init SUCCESS\n");

    printf("[Parent] Initial Refcount: %d (Expected: 0)\n", shm_refcnt());

    // Create child processes that attach
    for (i = 0; i < N; i++) {
        if (fork() == 0) {
            printf("[Child %d] Calling shm_attach()...\n", i);
            if (shm_attach() < 0) {
                printf("[Child %d] ERROR: shm_attach failed\n", i);
                exit(1);
            }

            printf("[Child %d] Attached | Refcount: %d\n", i, shm_refcnt());

            // Keep it attached for a while so parent can test destroy failure
            pause(20);

            printf("[Child %d] Calling shm_detach()...\n", i);
            if (shm_detach() < 0) {
                printf("[Child %d] ERROR: shm_detach failed\n", i);
                exit(1);
            }

            printf("[Child %d] Detached | Refcount: %d\n", i, shm_refcnt());
            exit(0);
        }

        pause(2);
    }

    // Try to destroy while children are still attached
    printf("\n[Parent] Attempting shm_destroy() while refcount > 0...\n");
    if (shm_destroy() < 0)
        printf("[Parent] CORRECT: shm_destroy FAILED (refcount not zero)\n");
    else
        printf("[Parent] ERROR: shm_destroy should not succeed!\n");

    // Wait for all children to finish (they will detach)
    for (i = 0; i < N; i++) wait(0);

    // Refcount should now be 0
    printf("\n[Parent] All children finished.\n");
    printf("[Parent] Current Refcount: %d (Expected: 0)\n", shm_refcnt());

    // Step 4: Now destroy should succeed
    printf("[Parent] Attempting shm_destroy() again...\n");
    if (shm_destroy() < 0)
        printf("[Parent] ERROR: shm_destroy failed unexpectedly\n");
    else
        printf("[Parent] SUCCESS: shm_destroy worked (refcount = 0)\n");

    exit(0);
}
