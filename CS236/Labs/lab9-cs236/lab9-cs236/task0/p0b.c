#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int total = 0;

void* increment(void *arg) {
    // TODO: Increment total by x for 1 million times, where x is the argument passed to the thread.
    for (int i = 0; i < 1000000; i++) {
        
    }
    // TODO: Return the total increment done by this thread (i.e., x * 1000000) as the thread's return value.
}

int main() {
    // TODO: Create threads to run increment function with arguments 1, 2, 3, and 4 respectively.

    // TODO: Wait for all threads to finish and print the total value.

    // printf("Thread %d incremented total by %d\n", _, _);
    // printf("Total: %d\n", total);
    return 0;
}