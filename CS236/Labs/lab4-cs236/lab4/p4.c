#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int main()
{
  int i;
  int pid;

  printf("Parent pid: %d\n", getpid());

  for (i = 0; i < 3; i++) {
    pid = fork();
    if (pid == 0) {
      pause(10);
      exit(0);
    }
  }

  pause(5);
  getProcessStates();
  pause(30);

  printf("\n=== Process states after exiting ===\n");
  getProcessStates();

  for (i = 0; i < 3; i++)
    wait(0);

  printf("\n=== Process states reaping ===\n");
  getProcessStates();

  exit(0);
}
