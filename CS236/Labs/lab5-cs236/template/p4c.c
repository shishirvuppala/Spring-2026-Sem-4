#include "kernel/types.h"
#include "user/user.h"

int main(int argc, char *argv[])
{
    int pid, upid;
    uint64 t1 = uptime();
    for(int i = 0; i < 100000; i++) {
         pid = getpid();
    }
    uint64 t2 = uptime();

    uint64 t3 = uptime();
    for(int i = 0; i < 100000; i++) {
         upid = ugetpid();
    }
    uint64 t4 = uptime();

    uint64 elapsed1 = t2 - t1;
    uint64 elapsed2 = t4 - t3;

    printf("PID from getpid() : %d time-taken: %lu\n", pid, elapsed1);
    printf("PID from ugetpid(): %d time-taken: %lu\n", upid, elapsed2);
    
    exit(0);
}
