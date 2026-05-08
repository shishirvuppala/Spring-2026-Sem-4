#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// do not modify anything in the template
int* createRandomArray(int);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s N\n", argv[0]);
        exit(1);
    }

    int N = atoi(argv[1]), *arr = createRandomArray(N);

    /* ---------------- WRITE YOUR CODE HERE ---------------- */

    /* Use the following print statements in your logic:
     * print in parent
     * printf("[pid %d] Created child process: %d\n", ___, ___);
     * printf("[pid %d] max(array): %d\n", ___, ___);
     * 
     * print in child processes
     * printf("[pid %d] arr[%d]: %d, Created child process: %d\n", ___, ___, ___, ___);
     * printf("[pid %d] arr[%d]: %d\n", ___, ___, ___]);
     * printf("[pid %d] arr[%d]: %d, child_max_est: %d, max_est: %d\n", ___, ___, ___, ___, ___);
     * printf("[pid %d] arr[%d]: %d, max_est: %d\n", ___, ___, ___, ___);
     */
}

int* createRandomArray(int N) {
    int* arr = malloc(N * sizeof(int));
    srand(time(NULL));

    // printf("[pid %d] Created array: ", ___);
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100;
        printf("%d ", arr[i]);
    }
    printf("\n");
    return arr;
}
