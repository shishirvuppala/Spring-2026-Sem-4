#include "kernel/types.h"
#include "user/user.h"

#define CYCLES 100000000  // how long to run (ticks)

// ------------------------------------------------------------

int busy() {
  int a = 0;
  for (int j = 0; j < 5; j++) {
    for (int i = 0; i < CYCLES; i++) {
      a += i;
      a %= 1000;
    }
  }
  pause(0);
  return a;
}

// ------------------------------------------------------------

void run_test(int use_priority) {
  if (use_priority) {
    setprioritysched(1);
  } else {
    setprioritysched(0);
  }
  // pause(10);  // let the scheduler stabilize

  int start = uptime();

  int pid1 = fork();
  if (pid1 == 0) {
    if (use_priority) {
      printf("[pid %d] setting high priority\n", getpid());
      setpriority(10);  // high priority
    }
    busy();
    int end = uptime();
    printf("[pid %d] start: %d, end: %d\n", getpid(),start, end);
    // printf("[pid %d] finished in %d ticks\n", getpid(), end - start);
    exit(0);
  }

  int pid2 = fork();
  if (pid2 == 0) {
    if (use_priority) {
      printf("[pid %d] setting low priority\n", getpid());
      setpriority(1);  // low priority
    }
    busy();
    int end = uptime();
    printf("[pid %d] start: %d, end: %d\n", getpid(),start, end);
    // printf("[pid %d] finished in %d ticks\n", getpid(), end - start);
    exit(0);
  }

  wait(0);
  wait(0);

  // cleanup

  setprioritysched(0);
}

// ------------------------------------------------------------

int main() {
  printf("=== Round Robin Scheduling ===\n");
  run_test(0);  // round robin

  printf("\n=== Priority Scheduling===\n");
  run_test(1);  // priority scheduling
  return 0;
}
