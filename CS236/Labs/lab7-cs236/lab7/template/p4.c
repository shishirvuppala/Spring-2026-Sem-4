#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

#define NCPU 8
#define RUN_TICKS 100
#define NUM_CHILDREN 4

int ins[NCPU], outs[NCPU];

void busy_loop(void) {
  int start = uptime();
  while (uptime() - start < RUN_TICKS) {
    for (volatile int i = 0; i < 1000000; i++);
  }
}

void print_stats(void) {
  printf("-> stats for child1: \n");
  getswitchstats(ins, outs);
  for (int i = 0; i < 2; i++) {
    printf("CPU %d: in=%d out=%d\n", i, ins[i], outs[i]);
  }
}

int main(void) {
  printf("=== Scenario: Without CPU Affinity ===\n");

  int pid = fork();
  if (pid == 0) {
    busy_loop();
    print_stats();
    exit(0);
  }
  for (int i = 0; i < NUM_CHILDREN - 1; i++) {
    pid = fork();
    if (pid == 0) {
      busy_loop();
      exit(0);
    }
  }
  for (int i = 0; i < NUM_CHILDREN; i++) {
    wait(0);
  }

  printf("\n=== Scenario: child1(CPU 0) others(CPU 1) ===\n");
  pid = fork();
  if (pid == 0) {
    setcpuaffinity(1);
    busy_loop();
    print_stats();
    exit(0);
  }
  for (int i = 0; i < NUM_CHILDREN - 1; i++) {
    pid = fork();
    if (pid == 0) {
      setcpuaffinity(2);
      busy_loop();
      exit(0);
    }
  }
  for (int i = 0; i < NUM_CHILDREN; i++) {
    wait(0);
  }


  printf("\n=== Scenario: child1(CPU 0 & 1) others(CPU 1) ===\n");
   pid = fork();
   if (pid == 0) {
     setcpuaffinity(3);
     busy_loop();
     print_stats();
     exit(0);
   }
   for (int i = 0; i < NUM_CHILDREN - 1; i++) {
     pid = fork();
     if (pid == 0) {
       setcpuaffinity(2);
       busy_loop();
       exit(0);
     }
   }
   for (int i = 0; i < NUM_CHILDREN; i++) {
     wait(0);
   }

  printf("\n=== Scenario: child1(CPU 0 & 1) others(CPU 0 & 1) ===\n");
  pid = fork();
  if (pid == 0) {
    setcpuaffinity(3);
    busy_loop();
    print_stats();
    exit(0);
  }
  for (int i = 0; i < NUM_CHILDREN - 1; i++) {
    pid = fork();
    if (pid == 0) {
      setcpuaffinity(3);
      busy_loop();
      exit(0);
    }
  }
  for (int i = 0; i < NUM_CHILDREN; i++) {
    wait(0);
  }
  return 0;
}
