#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int main()
{
  int pid = fork();
  if(pid == 0){
    pause(10);
    exit(0);
  }

  printf("Checking child pid %d\n", pid);
  areYouThere(pid);

  int mypid = getpid();
  printf("\nMy PID is %d\n", mypid);

  printf("Checking my own PID:\n");
  areYouThere(mypid);

  printf("\nChecking a non-existent PID (50)\n");
  areYouThere(50);

  exit(0);
}
