#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h> 
#include <unistd.h>
#include <time.h>
#include <stdint.h>

#define CLOCK_MONOTONIC 1

struct element{
    int64_t execution_time;
    struct timespec arrival_time;
};

int64_t max_ready_queue_size;     // maximum size of the ready queue
int64_t push_queue_counter = 0;   // critical_variable
int64_t pop_queue_counter = 0;    // critical_variable
int64_t queue_size = 0;           // critical_variable
int64_t num_cpus;                 // Number of CPU threads
struct element *ready_queue;      // Ready queue
char input_file[100];             // Input file name for the producer
volatile int64_t is_cpu_running = 1;

pthread_mutex_t ready_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t queue_has_work = PTHREAD_COND_INITIALIZER;

// metrics to track
int64_t *num_lock_reqs;           // to keep track of the number of times each thread requesting for the lock
int64_t *total_tasks;             // to count the total number of tasks/work done by each thread
int64_t reqs_dropped = 0;         // to count the number of request drops by producer when ready queue is full
int64_t *avg_lock_wait_time;      // calculating the average time for which each thread has to wait to get the lock
int64_t *max_task_wait_time;      // maximum waiting time of task to get scheduled(among all tasks)
int64_t *min_task_wait_time;      // minimum waiting time of task to get scheduled(among all tasks)
int64_t *avg_task_wait_time;      // average waiting time of each thread to dequeue the task from ready queue
int64_t avg_ready_queue_size = 0; // Average size of the ready queue

void* producer(void* args) {
    FILE* taskfile = fopen(input_file, "r");
    if (taskfile == NULL) {
        perror("Failed to open the task file");
        exit(1);
    }
    
    int64_t total_work;
    fscanf(taskfile, "%ld", &total_work);
    
    int64_t num_of_work, execution_time, next_arrival_time;
    while(total_work--) {
        fscanf(taskfile, "%ld %ld %ld", &num_of_work, &execution_time, &next_arrival_time);
        
        for(int64_t i=0;i<num_of_work;i++) {
            pthread_mutex_lock(&ready_queue_mutex);
            
            // Drop the request if the queue is full
            if(queue_size == max_ready_queue_size) {
                reqs_dropped++;
                pthread_mutex_unlock(&ready_queue_mutex);
                continue;
            }
            
            ready_queue[push_queue_counter].execution_time = execution_time;
            queue_size++;
            clock_gettime(CLOCK_MONOTONIC, &ready_queue[push_queue_counter].arrival_time); 
            push_queue_counter = (push_queue_counter + 1) % max_ready_queue_size;
            pthread_cond_signal(&queue_has_work);
            pthread_mutex_unlock(&ready_queue_mutex);
        }
        usleep(next_arrival_time);
    }
    pthread_exit(NULL);
}

void* cpu(void* args) {
    int64_t cpu_id = *((int*)args);
    free(args);

    struct timespec start, end;

    while(is_cpu_running == 1) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        pthread_mutex_lock(&ready_queue_mutex);
        clock_gettime(CLOCK_MONOTONIC, &end);

        num_lock_reqs[cpu_id]++;

        avg_lock_wait_time[cpu_id] += (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec)/1000;
        while (queue_size == 0) {
            pthread_cond_wait(&queue_has_work, &ready_queue_mutex);
            if(is_cpu_running == 0) {
                pthread_mutex_unlock(&ready_queue_mutex);
                goto out;
            }
        }
        int64_t execution_time = ready_queue[pop_queue_counter].execution_time;
        start = ready_queue[pop_queue_counter].arrival_time;
        ready_queue[pop_queue_counter].execution_time = -1;
        pop_queue_counter = (pop_queue_counter + 1) % max_ready_queue_size;
        queue_size--;
        total_tasks[cpu_id]++;
        pthread_mutex_unlock(&ready_queue_mutex);

        clock_gettime(CLOCK_MONOTONIC, &end);   // starting time of the task
        int64_t waiting_time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec)/1000;
        // printf("Waiting time of each task %ld\n",waiting_time);
        avg_task_wait_time[cpu_id] += waiting_time;
        if(waiting_time > max_task_wait_time[cpu_id]) max_task_wait_time[cpu_id] = waiting_time;
        if(waiting_time < min_task_wait_time[cpu_id]) min_task_wait_time[cpu_id] = waiting_time;
        

        // execute the process
        usleep(execution_time);
    }
out:
    if (num_lock_reqs[cpu_id] > 0)
        avg_lock_wait_time[cpu_id] /= num_lock_reqs[cpu_id];

    if (total_tasks[cpu_id] > 0)
        avg_task_wait_time[cpu_id] /= total_tasks[cpu_id];

    pthread_exit(NULL);
}

void* ready_queue_size_tracker(void* args) {
    int64_t counter = 0, size_tracker = 0;
    while (is_cpu_running == 1) {
        pthread_mutex_lock(&ready_queue_mutex);
        size_tracker += queue_size;
        pthread_mutex_unlock(&ready_queue_mutex);
        counter++;
        usleep(10);
    }
    if (counter > 0)
        avg_ready_queue_size = (size_tracker / counter);
    pthread_exit(NULL);
}

int main(int64_t argc, char* argv[]) {
    if (argc != 4) {
        perror("Execution syntax: ./task2 <file_name> <cpu_count> <max_ready_queue_size>");
        exit(0);
    }

    strcpy(input_file, argv[1]);
    max_ready_queue_size = atoi(argv[3]);
    num_cpus = atoi(argv[2]);
    if (num_cpus <= 0) {
        perror("Invalid CPU count");
        exit(0);
    }
    ready_queue = malloc(max_ready_queue_size * sizeof(struct element));
    for(int64_t i=0; i<max_ready_queue_size; i++) ready_queue[i].execution_time = -1;

    num_lock_reqs = malloc(num_cpus * sizeof(int64_t));
    total_tasks=malloc(num_cpus * sizeof(int64_t));
    avg_lock_wait_time = malloc(num_cpus * sizeof(int64_t));
    max_task_wait_time = malloc(num_cpus * sizeof(int64_t));
    min_task_wait_time = malloc(num_cpus * sizeof(int64_t));
    avg_task_wait_time = malloc(num_cpus * sizeof(int64_t));

    reqs_dropped=0;
    for(int64_t i=0; i<num_cpus; i++) {
        num_lock_reqs[i] = 0;
        total_tasks[i]=0;
        avg_lock_wait_time[i] = 0;
        max_task_wait_time[i] = 0;
        min_task_wait_time[i] = INT64_MAX;
        avg_task_wait_time[i] = 0;
    }

    pthread_t producer_thread, queue_size_tracker_thread;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    if (pthread_create(&queue_size_tracker_thread, NULL, ready_queue_size_tracker, NULL) != 0) {
        perror("Failed to create queue size tracker thread");
        exit(1);
    }

    if (pthread_create(&producer_thread, NULL, producer, NULL) != 0) {
        perror("Failed to create producer thread");
        exit(1);
    }

    pthread_t cpu_threads[num_cpus];
    for (int64_t i=0; i<num_cpus; i++) {
        int* cpu_id = (int*)malloc(sizeof(int));
        *cpu_id = i;
        if(pthread_create(&cpu_threads[i], NULL, cpu, cpu_id) != 0) {
            perror("Failed to create cpu thread");
            exit(1);
        }
    }

    pthread_join(producer_thread, NULL);
    
    while(1) {
        pthread_mutex_lock(&ready_queue_mutex);
        if(queue_size == 0) {
            pthread_mutex_unlock(&ready_queue_mutex);
            break;
        }
        pthread_mutex_unlock(&ready_queue_mutex);
        usleep(10);
    }

    is_cpu_running = 0;
    pthread_cond_broadcast(&queue_has_work);

    pthread_join(queue_size_tracker_thread, NULL);

    for (int64_t i=0; i<num_cpus; i++) {
        if (pthread_join(cpu_threads[i], NULL) != 0) {
            perror("Failed to join cpu thread");
            exit(1);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    int64_t total_execution_time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec)/1000;

    pthread_mutex_destroy(&ready_queue_mutex);
    pthread_cond_destroy(&queue_has_work);
    double graph_use=0;
    for(int64_t i=0; i<num_cpus; i++) {
        if (total_tasks[i] == 0) {
            min_task_wait_time[i] = 0;
        }

        graph_use+=avg_lock_wait_time[i];
        printf("CPU ID : %ld\n", i);
        printf("Number of lock requests : %ld \n", num_lock_reqs[i]);
        printf("Number of Tasks done: %ld \n", total_tasks[i]);
        printf("Average waiting time for lock : %ld us\n", avg_lock_wait_time[i]);
        printf("Max waiting time of a task : %ld us\n", max_task_wait_time[i]);
        printf("Min waiting time of a task : %ld us\n", min_task_wait_time[i]);
        printf("Average waiting time of a task : %ld us\n", avg_task_wait_time[i]);
        printf("\n");
    }
    graph_use/=num_cpus;
    printf("Number of drops done: %ld\n", reqs_dropped);
    printf("Average size of ready queue : %ld\n", avg_ready_queue_size);
    printf("Total execution time for all the requests : %ld us\n", total_execution_time);
    printf("Average waiting time for all threads %f us\n",graph_use);

    return 0;
}