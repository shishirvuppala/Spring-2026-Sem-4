#include "kernel/types.h"
#include "kernel/fcntl.h"
#include "user/user.h"

/*

This testcase checks if a newly created process is assigned the minimum virtual time out of all 
runnable/running processes

*/

#define NUM_PROCESSES 2

int abs(const int x){
  return x >= 0 ? x : -x;
}

// Work for "ticks" tick changes
void work(const int ticks){
    int workDone = 0;
    int prevTick = uptime();
    while(workDone < ticks){
        int currTick = uptime();
        if (currTick != prevTick){
            prevTick = currTick;
            workDone++;
        }
    }
}

int
test(const int wait_time)
{ 
    int id;
    int pids[NUM_PROCESSES + 1] = {0};
    pids[1] = getpid();

    // Fork a chain of processes.
    for(id = 1; id < NUM_PROCESSES && !fork(); id++){
      pids[id + 1] = getpid();
      work(1); // Reduce initial collisions between schedulers.
    }

    work(id * wait_time);

    // Parent reads the vruntime of itself and its children. 
    // Their minimum is the expected vruntime of the child.

    if (id == NUM_PROCESSES){
      int expected_vruntime = read_runtime(pids[1]);
      for(int i = 1; i <= NUM_PROCESSES; i++){
        const int vruntime = read_runtime(pids[i]);
        expected_vruntime = expected_vruntime > vruntime ? vruntime : expected_vruntime;
      }
      const int child = fork();
      if (!child)
        exit(0);
      const int actual_vruntime = read_runtime(child);
      const int result = abs(expected_vruntime - actual_vruntime) <= 1 && (expected_vruntime != 0);
      printf("Actual virtual runtime: %d, Expected virtual runtime: %d\n", actual_vruntime, expected_vruntime);
      exit(result);
    }

    int status;
    while(waitnohang(&status) == -2);
    if (id == 1)
      return status;
    exit(status);
    
}

int
main()
{
  printf("================================\n");
  printf("\tTesting fork\n");
  printf("================================\n");

  const int wait_times[] = {57};
  const int num_tests = sizeof(wait_times) / sizeof(int);
  int passed = 0;
  for(int i = 0; i < num_tests; i++){
    passed += test(wait_times[i]);
  }

  if (passed < num_tests) {
    printf("================================\n");
    printf("\tResult: [%d / %d]\n", passed, num_tests);
    printf("\ttc-fork FAILED\n");
    printf("================================\n");
  } 
  else {
    printf("================================\n");
    printf("\tResult: [%d / %d]\n", passed, num_tests);
    printf("\ttc-fork PASSED\n");
    printf("================================\n");
  }
  int result;
  for(int i = 0; i < num_tests; i++)
    wait(&result);
}
