#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  int p1_to_p2[2]; // Pipe: Parent -> Child 1
  int p2_to_p3[2]; // Pipe: Child 1 -> Child 2
  int p3_to_p1[2]; // Pipe: Child 2 -> Parent

  pid_t pid1, pid2;
  int num;

  // 1. Take num input from command line
  printf("Enter a Number: ");
  scanf("%d", &num);

  // 2. Create all three pipes
  if (pipe(p1_to_p2) == -1) {
    perror("pipe p1_to_p2");
    exit(1);
  }
  if (pipe(p2_to_p3) == -1) {
    perror("pipe p2_to_p3");
    exit(1);
  }
  if (pipe(p3_to_p1) == -1) {
    perror("pipe p3_to_p1");
    exit(1);
  }

  // 3. Create first child process (Child 1 / Process 2)
  pid1 = fork();
  if (pid1 == -1) {
    perror("fork pid1");
    exit(1);
  }

  if (pid1 == 0) {
    // ---- Child 1 (Process 2) ----
    // Close unused pipe ends
    close(p1_to_p2[1]); // Close write end of pipe from parent
    close(p2_to_p3[0]); // Close read end of pipe to child 2
    close(p3_to_p1[0]); // Close read end of pipe to parent
    close(p3_to_p1[1]); // Close write end of pipe to parent (not used by child 1)

    // Read number from parent
    int received;
    read(p1_to_p2[0], &received, sizeof(received));
    close(p1_to_p2[0]);

    printf("Child 1 received: %d\n", received);

    // Multiply by 2 and send to Child 2
    int result = received * 2;
    write(p2_to_p3[1], &result, sizeof(result));
    close(p2_to_p3[1]);

    exit(0);
  }

  // 4. Create second child process (Child 2 / Process 3)
  pid2 = fork();
  if (pid2 == -1) {
    perror("fork pid2");
    exit(1);
  }

  if (pid2 == 0) {
    // ---- Child 2 (Process 3) ----
    // Close unused pipe ends
    close(p1_to_p2[0]); // Close read end of pipe from parent (not used by child 2)
    close(p1_to_p2[1]); // Close write end of pipe from parent (not used by child 2)
    close(p2_to_p3[1]); // Close write end of pipe from child 1
    close(p3_to_p1[0]); // Close read end of pipe to parent

    // Read number from Child 1
    int received;
    read(p2_to_p3[0], &received, sizeof(received));
    close(p2_to_p3[0]);

    printf("Child 2 received: %d\n", received);

    // Multiply by 2 and send to Parent
    int result = received * 2;
    write(p3_to_p1[1], &result, sizeof(result));
    close(p3_to_p1[1]);

    exit(0);
  }

  // ---- Parent Process (Process 1) ----
  // Close unused pipe ends
  close(p1_to_p2[0]); // Close read end of pipe to child 1
  close(p2_to_p3[0]); // Close read end of pipe between children
  close(p2_to_p3[1]); // Close write end of pipe between children
  close(p3_to_p1[1]); // Close write end of pipe from child 2

  // Send integer to Child 1
  write(p1_to_p2[1], &num, sizeof(num));
  close(p1_to_p2[1]);

  // Wait for both children to finish
  wait(NULL);
  wait(NULL);

  // Read final result from Child 2
  int final_result;
  read(p3_to_p1[0], &final_result, sizeof(final_result));
  close(p3_to_p1[0]);

  // Print the result
  printf("Parent received: %d\n", final_result);

  return 0;
}