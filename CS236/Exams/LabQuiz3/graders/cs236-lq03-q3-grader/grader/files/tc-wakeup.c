// int abs(const int x){
//   return x >= 0 ? x : -x;
// }

// int
// test(const int wait_time)
// { 
//     const int parent_pid = getpid();
//     const int id = fork();
//     if (!id){
//         // Sleep for wait_time ticks.
//         const int child_pid = getpid();
//         pause(wait_time);
//         // Get the vruntime of the child and parent processes. 
//         // Since the parent is the only running process, when the child wakes up, its vruntime 
//         // should be equal to its parent's vruntime.
//         const int expected_vruntime = read_runtime(parent_pid);
//         const int actual_vruntime = read_runtime(child_pid);

//         printf("Actual virtual runtime: %d, Expected virtual runtime: %d\n", 
//                 actual_vruntime, expected_vruntime);
//         exit(abs(expected_vruntime - actual_vruntime) <= 5);
//     }
//     int status;
//     while(waitnohang(&status) == -2);
//     return status;
// }


#include "kernel/types.h"
#include "user/user.h"

/*

This testcase checks if a woken up process is assigned the minimum virtual time out of all 
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

    if (id == NUM_PROCESSES){
      const int child = fork();
      
      // Child reads the vruntime of all other processes. 
      // Their minimum is the expected vruntime of the child after waking up.
      if (!child){
        int expected_vruntime = read_runtime(pids[1]);
        for(int i = 1; i <= NUM_PROCESSES; i++){
          const int vruntime = read_runtime(pids[i]);
          expected_vruntime = expected_vruntime > vruntime ? vruntime : expected_vruntime;
        }
        pause(4);
        const int actual_vruntime = read_runtime(getpid());
        const int result = abs(expected_vruntime - actual_vruntime) <= 1 && (expected_vruntime != 0);
        printf("Actual virtual runtime: %d, Expected virtual runtime: %d\n", actual_vruntime, expected_vruntime);
        exit(result);
      }

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
  printf("\tTesting wakeup\n");
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
    printf("\ttc-wakeup FAILED\n");
    printf("================================\n");
  } 
  else {
    printf("================================\n");
    printf("\tResult: [%d / %d]\n", passed, num_tests);
    printf("\ttc-wakeup PASSED\n");
    printf("================================\n");
  }
  int result;
  for(int i = 0; i < num_tests; i++)
    wait(&result);
}