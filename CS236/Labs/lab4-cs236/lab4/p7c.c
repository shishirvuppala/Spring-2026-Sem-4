#include "kernel/types.h"
#include "kernel/stat.h"
#include "kernel/fcntl.h"
#include "user/user.h"



int
main(int argc, char *argv[])
{
//   int fd, i;
  strace_start();
    // printf("Creating a pipe\n");
    int p[2];
    if (pipe(p) < 0) {
        printf("pipe error\n");
        exit(1);
    }
    getpid();
    if(fork() != 0){
        exit(0);
    };
    wait(0);
//   strace_end();
  exit(0);
}
