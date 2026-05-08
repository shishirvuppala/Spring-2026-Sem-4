// p1a.c
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main() {
  int fd[2]; // fd[0] -> read end, fd[1] -> write end
  char buffer[100];
  char *message = "Hello World";

  // 1. Create the pipe using pipe()

  // 2. Write a message to the write end of the pipe

  // 3. Read the message from the read end of the pipe

  // 4. Print the message read from the pipe

  // 5. Close both ends of the pipe

  return 0;
}
