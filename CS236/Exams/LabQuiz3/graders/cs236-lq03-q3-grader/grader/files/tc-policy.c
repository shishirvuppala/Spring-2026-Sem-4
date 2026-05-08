#include "kernel/types.h"
#include "user/user.h"

#define MAXTICKS 500
#define NUM_PROCESSES 10
#define TIMESLICE 10

const int threshold = 1;

#define CHECK_TERMINATION(baseTick) if (uptime() > baseTick + MAXTICKS) break;

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

int* test(){
    int id;
    int* vruntimes = malloc(MAXTICKS * sizeof(int));
    int* minimumvruntimes = malloc(MAXTICKS * sizeof(int));
    int* ticks = malloc(MAXTICKS * sizeof(int));

    // Fork a chain of processes.
    for(id = 1; id < NUM_PROCESSES && !fork(); id++)
      work(1); // Reduce initial collisions between schedulers

    const int baseTick = uptime();
    int status;

    // Let the children work
    if (id != 1){
        int forkCount = 0;
        for(;;){
            CHECK_TERMINATION(baseTick);
            work(23);
            CHECK_TERMINATION(baseTick);
            pause(11);
            CHECK_TERMINATION(baseTick);
            work(19);
            CHECK_TERMINATION(baseTick);
            pause(7);
            CHECK_TERMINATION(baseTick);
            work(29);
            CHECK_TERMINATION(baseTick);
            fork();
            forkCount++;
        }

        myvruntimes(vruntimes);
        myminimumvruntimes(minimumvruntimes);
        myticks(ticks);
        int errors = 0;
        int checks = 0;

        while(waitnohang(&status) != -1){
            wait(&status);
            errors += status % 10000;
            checks += status / 10000;
        }

        int i = 0;
        while(i + 1 < MAXTICKS && ticks[i] && ticks[i + 1]){
            int executionStartTick = ticks[i];
            int executions = 1;
            if (vruntimes[i] != minimumvruntimes[i >> 1]){
                // printf("Scheduled process did not have the minimum vruntime: \n");
                // printf("\tMinimum vruntime: %d\n", minimumvruntimes[i >> 1]);
                // printf("\tActual vruntime: %d\n", vruntimes[i]);
                errors++;
            }
            checks++;
            while(1){
                if (i + 3 >= MAXTICKS || !ticks[i + 3])
                    break;
                if (ticks[i + 2] != ticks[i + 1])
                    break;
                if (ticks[i + 1] - executionStartTick >= TIMESLICE)
                    break;
                i+=2;
                executions++;
            }
            if (ticks[i + 1] - executionStartTick > TIMESLICE + 1){
                printf("Execution: %d %d %d\n", executionStartTick, ticks[i + 1], executions);
                printf("Timeslice Incorrect\n");
                errors = 1000;
                checks = 1000;
                break;
            }
            i+=2;
        }
        exit(checks * 10000 + errors);
    }

    else{
        int* result   = malloc(2 * sizeof(int));
        while(waitnohang(&status) == -2);
        result[0] = status % 10000;
        result[1] = status / 10000;
        return result;
    }
}


int main(){

    printf("=================================\n");
    printf("\tTesting policy\n");
    printf("=================================\n");

    int* result = test();
    const int percentage_error = (result[0] * 100) / result[1];

    if (result[0]){
    printf("=================================\n");
        printf("\tPolicy errors : %d%% (%d/%d)\n", percentage_error, result[0], result[1]);
        printf("\tThreshold : %d%%\n", threshold);
        if (percentage_error < threshold)
            printf("\ttc-policy PASSED\n");
        else
            printf("\ttc-policy FAILED\n");
    printf("=================================\n");
    }
    else{
    printf("=================================\n");
        printf("\tPolicy errors : 0%% (%d/%d)\n", result[0], result[1]);
        printf("\tThreshold : %d%%\n", threshold);
        printf("\ttc-policy PASSED\n");
    printf("=================================\n");
    }
}

