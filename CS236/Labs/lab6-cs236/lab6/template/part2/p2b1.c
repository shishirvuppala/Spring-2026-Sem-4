#include "kernel/types.h"
#include "user/user.h"

int
main(void)
{
    // 1. Attach without init
    printf("[Test 1] Attach without shm_init()...\n");
    if (shm_attach() == -1)
        printf("PASS: shm_attach() correctly failed\n\n");
    else
        printf("FAIL: shm_attach() should not succeed\n\n");

    // 2. Destroy without init
    printf("[Test 2] Destroy without shm_init()...\n");
    if (shm_destroy() == -1)
        printf("PASS: shm_destroy() correctly failed\n\n");
    else
        printf("FAIL: shm_destroy() should not succeed\n\n");

    // 3. Early detach
    printf("[Test 3] Detach before attach...\n");
    shm_init();
    if (shm_detach() == -1)
        printf("PASS: shm_detach() correctly failed before attach\n\n");
    else
        printf("FAIL: shm_detach() should not succeed\n\n");

    // 4. Multiple detaches in same process
    printf("[Test 4] Multiple detaches in same process...\n");
    if (shm_attach()) printf("First attach succeeded\n");
    if (shm_detach() == 0) printf("First detach succeeded\n");

    if (shm_detach() == -1)
        printf("PASS: Second detach correctly failed\n\n");
    else
        printf("FAIL: Second detach should not succeed\n\n");

    // 5. Attach twice in same process
    printf("[Test 5] Attach twice in same process...\n");
    if (shm_attach())
        printf(
            "PASS: Re-attach succeeded (should allow attach after detach)\n");

    if (shm_attach() == -1)
        printf("PASS: Second attach correctly failed\n\n");
    else
        printf("FAIL: Second attach should not succeed\n\n");

    printf("Detaching shared memory... \n");
    if (shm_detach() == 0)
        printf("SUCCESS\n\n");
    else
        printf("ERROR\n\n");

    // 6. Destroy while another process is attached
    printf("[Test 6] Destroy while child attached...\n");
    if (fork() == 0) {
        // Child
        if (shm_attach()) printf("[Child] Attached to shared memory\n");

        pause(7);  // allow parent to attempt destroy
        if (shm_detach() == 0) printf("[Child] Detached successfully\n");

        exit(0);
    } else {
        // Parent
        pause(2);  // give child time to attach
        
        if (shm_destroy() == -1)
            printf(
                "[Parent] PASS: shm_destroy() failed while child attached\n");
        else
            printf("[Parent] FAIL: shm_destroy() should not succeed\n");

        wait(0);
        if (shm_destroy() == 0)
            printf(
                "[Parent] PASS: shm_destroy() succeeded after child "
                "detached\n");
    }

    exit(0);
}
