#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int total = 0;

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void* increment(void *arg) {
    int x = *(int *)arg, *inc = calloc(1, sizeof(int));
    free(arg);
    for (int i = 0; i < 1000000; i++) {
        *inc += x;
    }
    pthread_mutex_lock(&lock);
    total += *inc;
    pthread_mutex_unlock(&lock);
    return inc;
}

#define N 4

int main() {
    pthread_t threads[N];
    for (int i = 0; i < N; i++) {
        int *arg = malloc(sizeof(int));
        *arg = i + 1;
        pthread_create(&threads[i], NULL, (void *)increment, arg);
    }
    for (int i = 0; i < N; i++) {
        int *ret;
        pthread_join(threads[i], (void **)&ret);
        printf("Thread %d incremented total by %d\n", i + 1, *ret);
        free(ret);
    }
    printf("Total: %d\n", total);

    pthread_mutex_destroy(&lock);
    return 0;
}