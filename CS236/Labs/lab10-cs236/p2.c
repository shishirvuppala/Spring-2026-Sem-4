#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int lock = 0;

int VAR = 0;

#define ITER 100000

void increment(int* thread_rank) {
  int tid = *thread_rank;
  free(thread_rank);

  for (int i = 0; i < ITER; i++) {
    // TODO: Uncomment this line after implement futex(_, FUTEX_LOCK, 1)
    futex(&lock, FUTEX_LOCK, 1);
    
    if (i > 0 && i % (ITER / (10)) == 0) {
      printf("[T-%d] %d/%d iterations done\n", tid, i, ITER);
    }
    
    int tmp;
    tmp = VAR;
    tmp -= 5;
    VAR = tmp;

    
    tmp = VAR;
    tmp += 2;
    VAR = tmp;

    tmp = VAR;
    tmp += 4;
    VAR = tmp;

    // TODO: Uncomment this line after implement futex(_, FUTEX_UNLOCK, 1)
    futex(&lock, FUTEX_UNLOCK, 1);
  }
  // TODO: Uncomment this line after implement futex(_, FUTEX_LOCK, 1)
    futex(&lock, FUTEX_LOCK, 1);
  printf("[T-%d] Finished\n", tid);
  // TODO: Uncomment this line after implement futex(_, FUTEX_UNLOCK, 1)
    futex(&lock, FUTEX_UNLOCK, 1);

  exit(0);
}

int main(int argc, char* argv[]) {
  int N = 4;
  printf("Calling Process Print VAR value: %d, N: %d\n", VAR, N);

  for (int i = 0; i < N; i++) {
    int* thread_rank = malloc(sizeof(int));
    *thread_rank = i;
    create_thread(increment, thread_rank);
  }

  for (int i = 0; i < N; i++) {
    join();
  }

  printf("All threads joined, VAR value: %d, expected: %d\n", VAR, N * ITER);
  exit(0);
}
