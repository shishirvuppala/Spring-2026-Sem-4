#include "kernel/types.h"
#include "user/user.h"

#define NCPU 8
#define RUNTIME 20

struct stats {
  char label[16];
  int pid;
  int in;
  int out;
};

// ------------------------------------------------------------

void collect_stats(struct stats* s, char* label) {
  int ins[NCPU];
  int outs[NCPU];
  int i;

  s->pid = getpid();

  // copy label
  for (i = 0; i < 15 && label[i]; i++) s->label[i] = label[i];
  s->label[i] = 0;

  if (getswitchstats(ins, outs) < 0) {
    s->in = -1;
    s->out = -1;
    return;
  }

  s->in = 0;
  s->out = 0;

  for (i = 0; i < NCPU; i++) {
    s->in += ins[i];
    s->out += outs[i];
  }
}

// ------------------------------------------------------------

void busy_worker(int fd, char* label) {
  int start = uptime();
  while (uptime() - start < RUNTIME)
    for (volatile int i = 0; i < 10000; i++);

  struct stats s;
  collect_stats(&s, label);

  write(fd, &s, sizeof(s));
  close(fd);
  exit(0);
}

void sleep_worker(int fd, char* label) {
  int start = uptime();
  while (uptime() - start < RUNTIME) pause(RUNTIME);

  struct stats s;
  collect_stats(&s, label);

  write(fd, &s, sizeof(s));
  close(fd);
  exit(0);
}

// ------------------------------------------------------------

void run_pair(int busy1, int busy2) {
  int p1[2], p2[2];
  pipe(p1);
  pipe(p2);

  if (fork() == 0) {
    close(p1[0]);
    if (busy1)
      busy_worker(p1[1], "busy-A");
    else
      sleep_worker(p1[1], "sleep-A");
  }

  if (fork() == 0) {
    close(p2[0]);
    if (busy2)
      busy_worker(p2[1], "busy-B");
    else
      sleep_worker(p2[1], "sleep-B");
  }

  close(p1[1]);
  close(p2[1]);

  struct stats s1, s2;

  read(p1[0], &s1, sizeof(s1));
  read(p2[0], &s2, sizeof(s2));

  close(p1[0]);
  close(p2[0]);

  wait(0);
  wait(0);

  printf("%s (pid=%d) in=%d out=%d\n", s1.label, s1.pid, s1.in, s1.out);

  printf("%s (pid=%d) in=%d out=%d\n", s2.label, s2.pid, s2.in, s2.out);
}
void run_single(int busy1) {
  int p1[2];
  pipe(p1);


  if (fork() == 0) {
    close(p1[0]);
    if (busy1)
      busy_worker(p1[1], "busy-A");
    else
      sleep_worker(p1[1], "sleep-A");
  }

  close(p1[1]);


  struct stats s1;

  read(p1[0], &s1, sizeof(s1));

  close(p1[0]);

  wait(0);
  printf("%s (pid=%d) in=%d out=%d\n", s1.label, s1.pid, s1.in, s1.out);
}



int main() {
  printf("------------------------------------\n");
  printf("RUNNING: single busy process for %d ticks\n", RUNTIME);
  run_single(1);

  printf("------------------------------------\n");
  printf("RUNNING: single sleep process for %d ticks\n", RUNTIME);
  run_single(0);

  printf("------------------------------------\n");
  printf("RUNNING: two BUSY processes for %d ticks\n", RUNTIME);
  run_pair(1, 1);

  printf("------------------------------------\n");
  printf("RUNNING: BUSY + SLEEP process for %d ticks\n", RUNTIME);
  run_pair(1, 0);

  printf("------------------------------------\n");
  printf("RUNNING: two SLEEP processes for %d ticks\n", RUNTIME);
  run_pair(0, 0);

  printf("------------------------------------\n");
  return 0;
}
