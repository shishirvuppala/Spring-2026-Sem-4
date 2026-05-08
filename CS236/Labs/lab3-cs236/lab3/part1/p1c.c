#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
  int p1_to_p2[2]; // Pipe: Parent -> Child 1
  int p2_to_p3[2]; // Pipe: Child 1 -> Child 2
  int p3_to_p1[2]; // Pipe: Child 2 -> Parent

  pid_t pid1, pid2;
  int num;

  // 1. take num input from command line

  // 2. Create all three pipes

  // 3. Create first child process (Child 1)

  // 4. Create second child process (Child 2)

  // ---- Parent Process (Process 1) ----
  // - Send integer to Child 1
  // - Read final result from Child 2
  // - Print the result

  return 0;
}
