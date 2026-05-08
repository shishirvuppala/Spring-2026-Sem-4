#include "kernel/types.h"
#include "kernel/fcntl.h"
#include "user/user.h"

// we don't read from this, so the
// block corresponding to this shouldn't
// be in the block cache
int main() {
  int fd = open("ls", O_RDONLY);
  if (fd == -1) {
    printf("opening failed\n");
    exit(1);
  }

  fstrace(fd, 100000);

  close(fd);
}
