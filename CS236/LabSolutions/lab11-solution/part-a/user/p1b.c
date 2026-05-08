#include "kernel/types.h"
#include "kernel/fcntl.h"
#include "user/user.h"

// both parent, child open the same file.
// struct file would be different, but both
// would be pointing to the same inode, which
// will have reference count 2 (if both of them
// have file opened at fstrace call)
int main() {
  fork();

  int fd = open("ls", O_RDONLY);
  if (fd == -1) {
    printf("opening failed\n");
    exit(1);
  }

  char buf[10];
  read(fd, buf, 10);

  pause(2); // so both parent, child have opened it by this
  fstrace(fd, -1);

  close(fd);
}
