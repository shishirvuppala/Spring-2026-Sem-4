#include "kernel/types.h"
#include "kernel/fcntl.h"
#include "user/user.h"

// two different fds pointing to the same file struct
// read/writes using one fd also effect read/writes using
// the other fd.
int main() {
  int fd = open("ls", O_RDONLY);
  if (fd == -1) {
    printf("opening failed\n");
    exit(1);
  }

  char buf[10];
  read(fd, buf, 10);

  int fd2 = dup(fd);

  fstrace(fd, -1);
  fstrace(fd2, -1);

  close(fd);
}
