#include<stdio.h>
#include<pthread.h>
#include<unistd.h>
#include<stdlib.h>
#include<sys/time.h>

int array_size, indices_per_thread;
int *A, *B, *C;
int threads_completed = 0;
int num_threads;

void* addArrays(void* arg){
    int start_index = *(int*)arg; // Pass this from the main thread
    // =============================================================
    // BEGIN YOUR CODE HERE
    // implement this function to perform addition of two arrays A and B, 
    // via multiple threads for their assigned indices and 
    // store the result in array C.







    // =============================================================

    // execution time added to simulate other work for threads
    // the main thread need not wait so long to know that work is done here
    usleep(random()%1000000);  
   
    pthread_exit(NULL);
}

int main(int argc, char* argv[]){

    struct timeval start_time, end_time;

    if(argc != 4) {
        perror("usage: ./task1 <file_name1> <file_name2> <number_of_parallel_threads>");
        exit(0);
    }

    FILE *file1, *file2;
    file1 = fopen(argv[1], "r");
    file2 = fopen(argv[2], "r");
    if(file1 == NULL || file2 == NULL) {
        perror("Error opening the file");
        exit(0);
    }
    fscanf(file1, "%d", &array_size);
    fscanf(file2, "%d", &array_size);

    num_threads = atoi(argv[3]);

    A = malloc(array_size * sizeof(int));
    B = malloc(array_size * sizeof(int));
    C = malloc(array_size * sizeof(int));

    for (int i=0; i<array_size; i++) { 
        fscanf(file1, "%d", &A[i]);
        fscanf(file2, "%d", &B[i]);
    }

    fclose(file1);
    fclose(file2);

    gettimeofday(&start_time,NULL);
    // =============================================================
    // BEGIN YOUR CODE HERE
    // Create num_threads number of threads, each thread needs a start index
    // Use mutexes/condition variables where required
    // Calculate the sum in the main thread once all other threads have calculated
    // their totals
    // Print the total sum obtained and the execution time
    



    


    // =============================================================
    gettimeofday(&end_time,NULL);

    // elapsed time in micro-seconds
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000.0 +
                          (end_time.tv_usec - start_time.tv_usec);
   

    // check if all threads have exited
    // clean up the synchronization primitives
    return 0;
}
