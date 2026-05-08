#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int lock = 0;
int VAR = 0;

#define ITER 5

void worker(int* arg) {
  int tid = *arg;
  free(arg);

  for (int i = 0; i < ITER; i++) {
    futex(&lock, FUTEX_LOCK, 1);

    printf("[T-%d] in CS\n", tid);

    pause(10);

    VAR++;

    futex(&lock, FUTEX_UNLOCK, 1);
  }

  exit(0);
}

int main() {
  int N = 4;

  for (int i = 0; i < N; i++) {
    int* arg = malloc(sizeof(int));
    *arg = i;
    create_thread(worker, arg);
  }

  for (int i = 0; i < N; i++) join();

  printf("Final VAR: %d\n", VAR);
  exit(0);
}
