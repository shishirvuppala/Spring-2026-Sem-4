#include "kernel/types.h"
#include "user/user.h"

#define MAXTICKS 200
#define NUM_PROCESSES 3
#define TIMESLICE 10

/*

This testcase checks whether virtual runtime is updated correctly or not.
The policy-testcase will not be checked if you fail this testcase.

*/

int abs(int x){
  return x >= 0 ? x : -x;
}

int
test()
{

    int id;
    int ticks[MAXTICKS];
    int vRunTimes[MAXTICKS];

    // Fork a chain of processes.
    for(id = 0; id < NUM_PROCESSES && !fork(); id++);

    // Add base tick into ticks array.
    const int baseUpTime = uptime();

    // Create array of ticks and vruntimes when the process ran
    while(uptime() <= MAXTICKS + baseUpTime);
    myticks(ticks);
    myvruntimes(vRunTimes);

    // Process the array to check for correct updation of vruntime, only at the start of each execution.
    int numExecutions = 0;
    int i = 0;
    int errorno = 0;
    int totalRuntime = vRunTimes[0];
    int expectedVRunTime[MAXTICKS], observedVRunTime[MAXTICKS];

    while(i + 1 < MAXTICKS && ticks[i] && ticks[i + 1]){ 

        // Check vruntime before the start of this execution.
        int startTick = ticks[i];
        int endTick = ticks[i + 1];
        int startVRuntime = vRunTimes[i];

        if (abs(startVRuntime - totalRuntime) > 1){
          expectedVRunTime[errorno] = totalRuntime;
          observedVRunTime[errorno] = startVRuntime;
          errorno++;
        }
        totalRuntime += endTick - startTick;
        i += 2;
        numExecutions++;
    }

  int status;
  while(waitnohang(&status) == -2);

  for (int i = 0; i < errorno && i < 10; i++) {
    printf("[PROCESS ID %d] Error %d| Expected Virtual Runtime: %d, Observed Virtual Runtime: %d\n",
           id, i + 1, expectedVRunTime[i], observedVRunTime[i]);
  }

  if (errorno > 10) {
    printf(".\n.\n.\n\n");
  }
  else if (!errorno && numExecutions >= 2){
    printf("[PROCESS ID %d] %d virtual runtimes match\n", id, numExecutions - 1);
  }

  int result;

  if (status == 2 || numExecutions < 2)
    result = 2;
  else
    result = errorno || status;

  if (id == 0)
    return result;
  else
    exit(result);
}

int
main()
{
  printf("================================\n");
  printf("\tTesting vruntime\n");
  printf("================================\n");

  int result = test();
  if (result == 0) {
    printf("================================\n");
    printf("\ttc-vruntime PASSED\n");
    printf("================================\n");
  } else if (result == 1) {
    printf("================================\n");
    printf("Virtual Runtimes do not match\n");
    printf("\ttc-vruntime FAILED\n");
    printf("================================\n");
  } else {
    printf("================================\n");
    printf("Process not scheduled out\n");
    printf("\ttc-vruntime FAILED\n");
    printf("================================\n");
  }
  exit(0);
}
