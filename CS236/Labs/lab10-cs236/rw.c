#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

#define NREADERS 3
#define NWRITERS 2
#define NOPS 5

// readers only read shared_data
// writers increment shared_data
int shared_data = 0;

/* synchronization variables */
sem_t mutex;        // protects read_count
sem_t rw_mutex;     // controls access to shared resource

int read_count = 0;

int print_lock = 0;


// TODO: reads shared_data and prints the value
// acquire reader-lock before accessing shared_data
void read_shared_data(int tid) {
    // need print lock here
    // printf("R[%d]: read %d\n", tid, val);
}

// TODO: increments shared_data and prints the new value
// acquire writer-lock before accessing shared_data
void write_shared_data() {
    // No need for print lock here... why?
    // printf("W: wrote %d\n", val);
}


void reader(int *arg) {
    int tid = *arg;

    for (int i = 0; i < NOPS; i++) {
        read_shared_data(tid);
    }

    exit(0);
}

void writer(int *arg) {

    for (int i = 0; i < NOPS; i++) {
        write_shared_data();
    }

    exit(0);
}


int main(int argc, char *argv[]) {

    /* TODO: init semaphores */
    mutex.lock = rw_mutex.lock = 0;
    // sem_init(&mutex, __);
    // sem_init(&rw_mutex, __);

    int r0 = 0, r1 = 1, r2 = 2;
    int w0 = 0;

    create_thread(writer, &w0);

    create_thread(reader, &r0);
    create_thread(reader, &r1);
    create_thread(reader, &r2);

    for (int i = 0; i < 4; i++) {
        join();
    }

    printf("Final value of shared_data: %d\n", shared_data);

    exit(0);
}
