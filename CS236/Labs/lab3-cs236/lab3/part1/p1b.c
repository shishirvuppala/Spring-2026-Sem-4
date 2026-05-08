#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  int pipe1[2]; // Parent -> Child
  int pipe2[2]; // Child -> Parent
  pid_t pid;
  char buffer[20];

  // 1. Initialize both pipes

  pid = fork();

  if (pid > 0) {
    // PARENT PROCESS
    // 2. Write "Ping" to pipe1
    // 6. Read response from pipe2
    // 7. Print "[PID=process_id, PPID=parent_process_id] Parent received: ..." and close remaining ends

  } else {
    // CHILD PROCESS
    // 3. Read "Ping" from pipe1
    // 4. Print "[PID=process_id, PPID=parent_process_id] Child received: ..."
    // 5. Write "Pong" to pipe2 and close remaining ends
  }

  return 0;
}
