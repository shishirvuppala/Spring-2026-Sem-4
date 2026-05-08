// #include "kernel/types.h"
// #include "user/user.h"

// #define MAXTICKS 200
// #define NUM_PROCESSES 10


// /*

// This testcase checks whether a process scheduled once runs for 10 ticks or not.
// The vruntime testcase will not be checked if you fail this testcase. 

// */

// void test(){
//     // Fork 10 direct child processes
//     int id;
//     int currUpTime = 0;
//     int ticks[MAXTICKS];
//     int numTickChanges = 1;

//     for(id = 0; id < NUM_PROCESSES && fork(); id++);

//     // Parent returns
//     if (id == NUM_PROCESSES)
//         return;

//     // Add base tick into ticks array.
//     const int baseUpTime = uptime();
//     int prevUpTime = baseUpTime;
//     ticks[0] = baseUpTime;

//     // Create array of ticks when the process ran
//     while(1){
//         currUpTime = uptime();
//         if (currUpTime > MAXTICKS + baseUpTime)
//             break;

//         if (currUpTime != prevUpTime){
//             ticks[numTickChanges] = currUpTime;
//             prevUpTime = currUpTime;
//             numTickChanges++;
//         }
//     }

//     // Process the ticks, break down into contiguous tick groups, and infer the minimum timeslice
//     int numExecutions = 0;
//     int i = 0;
//     int left[MAXTICKS], right[MAXTICKS];
//     while(i < numTickChanges){ 
//         left[numExecutions] = ticks[i];
//         while(i + 1 < numTickChanges && ticks[i + 1] == ticks[i] + 1)
//             i++;
//         right[numExecutions] = ticks[i];
//         i++;
//         numExecutions++;
//     }
//     int minimum_slice = 1e9;
//     for(int i = 0; i < numExecutions; i++){
//         if (right[i] != baseUpTime + MAXTICKS){
//             minimum_slice = minimum_slice < right[i] - left[i] + 1 ? minimum_slice : right[i] - left[i] + 1;
//         }
//     }
//     if (numExecutions > 1)
//         exit(minimum_slice);
//     exit(0);
// }


// int main(){
//     printf("================================\n");
//     printf("\tTesting timeslice\n");
//     printf("================================\n");
//     test();
//     int failed = 0;
//     int zeroCount = 0;
//     for(int i = 0; i < NUM_PROCESSES; i++){
//         int status;
//         int pid = wait(&status);
//         if (status != 10 && status != 0)
//             failed = 1;
//         if (status == 0)
//             zeroCount++;
//         printf("[PID %d] executed for a minimum timeslice of %d\n", pid, status);
//     }
//     if (failed){
//         printf("================================\n");
//         printf("\ttc-slice FAILED\n");
//         printf("\tMinimum Timeslice is not 10\n");
//         printf("================================\n");
//     }
//     else if (zeroCount == NUM_PROCESSES){
//         printf("================================\n");
//         printf("\ttc-slice FAILED\n");
//         printf("\tNo process scheduled out\n");
//         printf("================================\n");
//     }
//     else{
//         printf("================================\n");
//         printf("\ttc-slice PASSED\n");
//         printf("================================\n");
//     }
// }

#include "kernel/types.h"
#include "user/user.h"

#define MAXTICKS 200
#define NUM_PROCESSES 3


/*

This testcase checks whether a process scheduled once runs for 10 ticks or not.
The vruntime testcase will not be checked if you fail this testcase. 

*/

void test(){
    // Fork 10 direct child processes
    int id;
    int ticks[MAXTICKS];

    for(id = 0; id < NUM_PROCESSES && fork(); id++);

    // Parent returns
    if (id == NUM_PROCESSES)
        return;

    // Add base tick into ticks array.
    const int baseUpTime = uptime();

    // Create array of ticks when the process ran
    while(uptime() <= MAXTICKS + baseUpTime);
    myticks(ticks);

    // Process the ticks, break down into contiguous tick groups, and infer the minimum timeslice
    int numExecutions = 0;
    int i = 0;
    int left[MAXTICKS], right[MAXTICKS];
    while(i + 1 < MAXTICKS && ticks[i] && ticks[i + 1]){ 
        int executionCompleted = 1;
        left[numExecutions] = ticks[i];
        while(1){
            if (i + 3 >= MAXTICKS || !ticks[i + 3]){
                executionCompleted = 0;
                break;
            }
            if (ticks[i + 2] != ticks[i + 1])
                break;
            i+=2;
        }
        if (executionCompleted){
            right[numExecutions] = ticks[i + 1];
            i+=2;
            numExecutions++;
        }
        else
            break;
    }
    int minimum_slice = 1e9;
    for(int i = 0; i < numExecutions; i++)
        minimum_slice = minimum_slice < right[i] - left[i] ? minimum_slice : right[i] - left[i];
    if (numExecutions > 1)
        exit(minimum_slice);
    exit(0);
}


int main(){
    printf("================================\n");
    printf("\tTesting timeslice\n");
    printf("================================\n");
    test();
    int failed = 0;
    int zeroCount = 0;
    for(int i = 0; i < NUM_PROCESSES; i++){
        int status;
        int pid = wait(&status);
        if (status != 10 && status != 0)
            failed = 1;
        if (status == 0)
            zeroCount++;
        printf("[PID %d] executed for a minimum timeslice of %d\n", pid, status);
    }
    if (failed){
        printf("================================\n");
        printf("\ttc-slice FAILED\n");
        printf("\tMinimum Timeslice is not 10\n");
        printf("================================\n");
    }
    else if (zeroCount == NUM_PROCESSES){
        printf("================================\n");
        printf("\ttc-slice FAILED\n");
        printf("\tNo process scheduled out\n");
        printf("================================\n");
    }
    else{
        printf("================================\n");
        printf("\ttc-slice PASSED\n");
        printf("================================\n");
    }
}
