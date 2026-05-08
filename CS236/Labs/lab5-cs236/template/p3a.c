#include "kernel/types.h"
#include "user/user.h"
#include "kernel/vm.h"

int
main(int argc, char *argv[])
{
  int vasize = getvasize();
  printf("---------------SBRK LAZY:---------------\n");
  printf("Virtual memory size before sbrk lazy : %d bytes\n", vasize);
  printf("Allocating 4096 bytes lazily...\n");
  sys_sbrk(4096 , SBRK_LAZY);
  vasize = getvasize();
  printf("Virtual memory size after sbrk lazy: %d bytes\n\n", vasize);

  printf("---------------SBRK EAGER:---------------\n");
  vasize = getvasize();
  printf("Virtual memory size before sbrk eager: %d bytes\n", vasize);
  printf("Allocating 4096 bytes eagerly...\n");
  sys_sbrk(4096 , SBRK_EAGER);
  vasize = getvasize();
  printf("Virtual memory size after sbrk eager: %d bytes\n", vasize);

  exit(0);
}
