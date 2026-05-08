#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

#define NBUF 2
#define NITEMS 4
int n_produce = -1, n_consume = -1;
/* shared buffer */
int buffer[NBUF];
int in = 0;
int out = 0;

/* semaphores */
sem_t mutex;
sem_t empty;
sem_t full;


// TODO
void produce(int item) {
    // puts item into buffer
}

int consume() {
    // removes item from buffer
    // and return it
}

int print_lock = 0;

void producer(int *arg)
{
    int tid = *arg;
    for (int i = 0; i < n_produce; i++) {

        /* TODO:
         * produce(tid * 10 + i);
         * printf("P[%d]: produced %d\n", tid, tid * 10 + i);
         */
    }

    exit(0);
}

void consumer(int *arg)
{
    int tid = *arg;
    for (int i = 0; i < n_consume; i++) {

        /* TODO:
         * int consumed = consume()
         * printf("C[%d]: consumed %d\n", tid, consumed);
         */
    }

    exit(0);
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: prod-cons <mode>\n");
        printf("mode: 0 = 1P1C, 1 = 2P1C, 2 = 3P3C\n");
        exit(0);
    }

    int mode = atoi(argv[1]);

    n_produce = n_consume = NITEMS;

    // TODO
    /* initialize semaphores */
    mutex.lock = empty.lock = full.lock = 0;
    sem_init(&mutex, __);
    sem_init(&empty, __);
    sem_init(&full, __);

    if (mode == 0) {
        int p = 0, c = 0;
        create_thread(producer, &p);
        create_thread(consumer, &c);

        for (int i = 0; i < 2; i++) {
            join();
        }
    }

    else if (mode == 1) {
	n_consume *= 2;
        int p0 = 0, p1 = 1, c = 0;
        create_thread(producer, &p0);
        create_thread(producer, &p1);
        create_thread(consumer, &c);

        for (int i = 0; i < 3; i++) {
            join();
        }
    }

    else if (mode == 2) {
        int p0 = 0, p1 = 1, p2 = 2, c0 = 0, c1 = 1, c2 = 2;
        create_thread(producer, &p0);
        create_thread(producer, &p1);
        create_thread(producer, &p2);
        create_thread(consumer, &c0);
        create_thread(consumer, &c1);
        create_thread(consumer, &c2);

        for (int i = 0; i < 6; i++) {
            join();
        }
    }

    else {
        printf("Invalid mode\n");
        exit(0);
    }

    exit(0);
}
