// p1a.c
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main() {
  int fd[2]; // fd[0] -> read end, fd[1] -> write end
  char buffer[100];
  char *message = "Hello World";

  // 1. Create the pipe using pipe()
  if (pipe(fd) == -1) {
    perror("pipe");
    return 1;
  }

  // 2. Write a message to the write end of the pipe
  printf("Write to pipe: %s\n", message);
  write(fd[1], message, strlen(message) + 1);

  // 3. Read the message from the read end of the pipe
  read(fd[0], buffer, sizeof(buffer));

  // 4. Print the message read from the pipe
  printf("Read from pipe: %s\n", buffer);

  // 5. Close both ends of the pipe
  close(fd[0]);
  close(fd[1]);

  return 0;
}