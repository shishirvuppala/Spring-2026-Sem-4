#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

int* createRandomArray(int);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s N\n", argv[0]);
        exit(1);
    }

    int N = atoi(argv[1]), *arr = createRandomArray(N), status, result;

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(1);
    }

    if (pid != 0) {
        printf("[pid %d] Created child process: %d\n", getpid(), pid);
        wait(&status);
        result = WEXITSTATUS(status);
        printf("[pid %d] max(array): %d\n", getpid(), result);
        exit(0);
    }

    for (int i = 0; i < N - 1; i++) {
        pid = fork();
        if (pid < 0) {
            perror("fork");
            exit(1);
        }
        if (pid != 0) {
            printf("[pid %d] arr[%d]: %d, Created child process: %d\n",
                getpid(), i, arr[i], pid);
            wait(&status);
            result = WEXITSTATUS(status);
            int exit_status = result > arr[i] ? result : arr[i];
            printf(
                "[pid %d] arr[%d]: %d, child_max_est: %d, "
                "max_est: %d\n",
                getpid(), i, arr[i], result, exit_status);
            exit(exit_status);
        }
    }
    printf("[pid %d] arr[%d]: %d\n", getpid(), N - 1, arr[N - 1]);
    printf("[pid %d] arr[%d]: %d, max_est: %d\n", getpid(), N - 1,
           arr[N - 1], arr[N - 1]);
    exit(arr[N - 1]);
}

int* createRandomArray(int N) {
    int* arr = malloc(N * sizeof(int));
    srand(time(NULL));

    printf("[pid %d] Created array: ", getpid());
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100;
        printf("%d ", arr[i]);
    }
    printf("\n");
    return arr;
}