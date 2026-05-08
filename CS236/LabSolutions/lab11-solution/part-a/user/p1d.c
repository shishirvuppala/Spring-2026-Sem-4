#include "kernel/types.h"
#include "kernel/fcntl.h"
#include "user/user.h"

// open syscall leads to creation of
// a new file struct, thus the offsets
// maintained in both would be different
// and read/writes using one fd doesn't
// effect the other
int main() {
  int fd = open("ls", O_RDONLY);
  if (fd == -1) {
    printf("opening failed\n");
    exit(1);
  }

  int fd2 = open("ls", O_RDONLY);
  if (fd2 == -1) {
    printf("opening failed\n");
    exit(1);
  }

  char buf[10];
  read(fd2, buf, 10);

  fstrace(fd, -1);
  fstrace(fd2, -1);

  close(fd);
}
