#include "kernel/types.h"
#include "user/user.h"

#define MAGIC_NUM 42

int
main(void)
{
    struct shm_channel* ch = (struct shm_channel*)kmap();
    if ((uint64)ch == -1) {
        printf("kmap failed\n");
        exit(1);
    }

    // ----- ECHO -----

    ch->u_cmd = CMD_ECHO;
    ch->u_ready = 1;

    while (ch->u_ack == 0);

    ch->u_ready = 0;
    ch->u_ack = 0;

    uint64 k_res = ch->k_res;
    ch->k_ack = 1;

    printf("Echoed: %d\n", (int)k_res);
    if (k_res != MAGIC_NUM) {
        printf("Unexpected result from kernel\n");
        exit(1);
    }

    // ----- GETPID -----

    ch->u_cmd = CMD_GETPID;
    ch->u_ready = 1;

    while (ch->u_ack == 0);

    ch->u_ready = 0;
    ch->u_ack = 0;

    printf("PID: %d\n", (int)ch->k_res);

    ch->k_ack = 1;

    // ----- GETPPID -----

    ch->u_cmd = CMD_GETPPID;
    ch->u_ready = 1;

    while (ch->u_ack == 0);
    
    ch->u_ready = 0;
    ch->u_ack = 0;

    printf("PPID: %d\n", (int)ch->k_res);

    ch->k_ack = 1;

    // ----- UPTIME -----

    ch->u_cmd = CMD_UPTIME;
    ch->u_ready = 1;

    while (ch->u_ack == 0);
    
    ch->u_ready = 0;
    ch->u_ack = 0;

    printf("Uptime: %d\n", (int)ch->k_res);

    ch->k_ack = 1;

    // ----- PROC_COUNT -----

    ch->u_cmd = CMD_PROC_COUNT;
    ch->u_ready = 1;

    while(ch->u_ack == 0);

    ch->u_ready = 0;
    ch->u_ack = 0;

    printf("No of Processes: %d\n", (int)ch->k_res);

    ch->k_ack = 1;

    exit(0);
}
