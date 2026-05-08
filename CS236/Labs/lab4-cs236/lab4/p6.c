#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int main()
{
  int i;
  int pid;

  printf("Initial child count: %d\n", getChildCount());

  for(i = 0; i < 3; i++){
    pid = fork();
    if(pid == 0){
      exit(0);
    }
  }

  printf("Child count after 3 forks: %d\n", getChildCount());

  for(i = 0; i < 3; i++){
    wait(0);
  }

  printf("Child count after reaping all childrens: %d\n", getChildCount());

  pid = fork();
  if(pid == 0){
    exit(0);
  }

  printf("Child count after 4th fork: %d\n", getChildCount());

  wait(0);
  wait(0);

  printf("Child count after reaping 4th child: %d\n", getChildCount());

  exit(0);
}
