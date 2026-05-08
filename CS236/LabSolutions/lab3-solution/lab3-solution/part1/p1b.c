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
  if (pipe(pipe1) == -1) {
    perror("pipe1");
    exit(1);
  }
  if (pipe(pipe2) == -1) {
    perror("pipe2");
    exit(1);
  }

  pid = fork();

  if (pid < 0) {
    perror("fork");
    exit(1);
  } else if (pid > 0) {
    // PARENT PROCESS
    // Close unused ends
    close(pipe1[0]); // Close read end of pipe1 (parent writes to pipe1)
    close(pipe2[1]); // Close write end of pipe2 (parent reads from pipe2)

    // 2. Write "Ping" to pipe1
    char *ping = "Ping";
    write(pipe1[1], ping, strlen(ping) + 1);
    close(pipe1[1]); // Close write end after writing

    // 6. Read response from pipe2
    read(pipe2[0], buffer, sizeof(buffer));

    // 7. Print "[PID=process_id, PPID=parent_process_id] Parent received: ..." and close remaining ends
    printf("[PID=%d, PPID=%d] Parent received: %s\n", getpid(), getppid(), buffer);
    close(pipe2[0]);

    // Wait for child to finish
    wait(NULL);

  } else {
    // CHILD PROCESS
    // Close unused ends
    close(pipe1[1]); // Close write end of pipe1 (child reads from pipe1)
    close(pipe2[0]); // Close read end of pipe2 (child writes to pipe2)

    // 3. Read "Ping" from pipe1
    read(pipe1[0], buffer, sizeof(buffer));
    close(pipe1[0]); // Close read end after reading

    // 4. Print "[PID=process_id, PPID=parent_process_id] Child received: ..."
    printf("[PID=%d, PPID=%d] Child received: %s\n", getpid(), getppid(), buffer);

    // 5. Write "Pong" to pipe2 and close remaining ends
    char *pong = "Pong";
    write(pipe2[1], pong, strlen(pong) + 1);
    close(pipe2[1]);
  }

  return 0;
}