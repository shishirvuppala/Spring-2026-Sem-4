#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h> 
#include <unistd.h>
#include <time.h>
#include <stdint.h>

#define CLOCK_MONOTONIC 1

// Define all the required synchronization variables
// TODO

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
int64_t is_cpu_running = 1;       // Flag that tells the CPU threads to keep on running, change it to exit the CPU threads

// metrics to track
int64_t *num_lock_reqs;           // to keep track of the number of times each thread requesting for the lock
int64_t *total_tasks;             // to count the total number of tasks/work done by each thread
int64_t reqs_dropped = 0;         // to count the number of request drops by producer when ready queue is full
int64_t *avg_lock_wait_time;      // calculating the average time for which each thread has to wait to get the lock
int64_t *max_task_wait_time;      // maximum waiting time of task to get scheduled(among all tasks)
int64_t *min_task_wait_time;      // minimum waiting time of task to get scheduled(among all tasks)
int64_t *avg_task_wait_time;      // average waiting time of each thread to dequeue the task from ready queue
int64_t avg_ready_queue_size = 0; // Average size of the ready queue


// Add all synchronization variables that are required in the producer code, and variables to track the metrics
// TODO
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
            // Drop the request if the queue is full
            if(ready_queue[push_queue_counter].execution_time != -1){
                reqs_dropped++;
                continue;
            }
            
            // Add work to the queue
            ready_queue[push_queue_counter].execution_time = execution_time;
            queue_size++;

            clock_gettime(CLOCK_MONOTONIC, &ready_queue[push_queue_counter].arrival_time); 
            push_queue_counter = (push_queue_counter + 1) % max_ready_queue_size;







        }
        usleep(next_arrival_time);
    }
    pthread_exit(NULL);
}


// Write the CPU code along with all the synchronization variables
// TODO
void* cpu(void* args) {
    int64_t cpu_id = *((int*)args);       // pass this argument from the main thread

    struct timespec start, end;

    while(is_cpu_running == 1) {
    







    }
}

// Write the code for tracking the ready queue after periodic intervals from this function (run as separate thread)
// TODO
void* ready_queue_size_tracker(void* args) {
    









}

// Write the logic for starting all the threads in a proper order, and gracefully exiting all the threads to print the metrics at the end
// TODO
int main(int64_t argc, char* argv[]) {
    if (argc != 4) {
        perror("Execution syntax: ./task2 <file_name> <cpu_count> <max_ready_queue_size>");
        exit(0);
    }

    strcpy(input_file, argv[1]);
    max_ready_queue_size = atoi(argv[3]);
    num_cpus = atoi(argv[2]);
    ready_queue = malloc(max_ready_queue_size * sizeof(struct element));
    for(int64_t i=0; i<max_ready_queue_size; i++) ready_queue[i].execution_time = -1;

    num_lock_reqs = malloc(num_cpus * sizeof(int64_t));
    total_tasks=malloc(num_cpus * sizeof(int64_t));;
    avg_lock_wait_time = malloc(num_cpus * sizeof(int64_t));
    max_task_wait_time = malloc(num_cpus * sizeof(int64_t));
    min_task_wait_time = malloc(num_cpus * sizeof(int64_t));
    avg_task_wait_time = malloc(num_cpus * sizeof(int64_t));

    for(int64_t i=0; i<num_cpus; i++) {
        num_lock_reqs[i] = 0;
        total_tasks[i]=0;
        reqs_dropped=0;
        avg_lock_wait_time[i] = 0;
        max_task_wait_time[i] = 0;
        min_task_wait_time[i] = 100000;
        avg_task_wait_time[i] = 0;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    























    clock_gettime(CLOCK_MONOTONIC, &end);
    int64_t total_execution_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec)/1000;

    double average_waiting_time_of_all_threads=0;
    for(int64_t i=0; i<num_cpus; i++) {
        average_waiting_time_of_all_threads+=avg_lock_wait_time[i];
        printf("CPU ID : %ld\n", i);
        printf("Number of lock requests : %ld \n", num_lock_reqs[i]);
        printf("Number of Tasks done: %ld \n", total_tasks[i]);
        printf("Average waiting time for lock : %ld us\n", avg_lock_wait_time[i]);
        printf("Max waiting time of a task : %ld us\n", max_task_wait_time[i]);
        printf("Min waiting time of a task : %ld us\n", min_task_wait_time[i]);
        printf("Average waiting time of a task : %ld us\n", avg_task_wait_time[i]);
        printf("\n");
    }
    average_waiting_time_of_all_threads/=num_cpus;
    printf("Number of drops done: %ld\n", reqs_dropped);
    printf("Average size of ready queue : %ld\n", avg_ready_queue_size);
    printf("Total execution time for all the requests : %ld us\n", total_execution_time);
    printf("Average waiting time for all threads %f\n",average_waiting_time_of_all_threads);

    return 0;
}
