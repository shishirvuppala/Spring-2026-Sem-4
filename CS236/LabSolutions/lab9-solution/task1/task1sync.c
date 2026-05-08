#include<stdio.h>
#include<pthread.h>
#include<unistd.h>
#include<stdlib.h>
#include<sys/time.h>


int array_size, indices_per_thread;
int *A, *B, *C;
int threads_completed = 0;
int num_of_threads;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

void* addArrays(void* arg){
    int start_index = *(int*)arg;
    int end_index = start_index + indices_per_thread;
    
    for(int i=start_index; i<end_index && i<array_size; i++){
        C[i] = A[i] + B[i];
    }
    
    pthread_mutex_lock(&mutex);
    threads_completed++;
    pthread_mutex_unlock(&mutex);

    if(threads_completed == num_of_threads)
        pthread_cond_signal(&cond);
    
    usleep(random()%1000000);//random execution time added
    pthread_exit(NULL);
}

int main(int argc, char* argv[]){
    if(argc != 4) {
        perror("Execution syntax: ./task1 <file_name1> <file_name2> <number_of_parallel_threads>");
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

    num_of_threads = atoi(argv[3]);
    indices_per_thread = (array_size % num_of_threads == 0) ? (array_size / num_of_threads) : (array_size / num_of_threads) + 1;

    A = malloc(array_size * sizeof(int));
    B = malloc(array_size * sizeof(int));
    C = malloc(array_size * sizeof(int));

    for (int i=0; i<array_size; i++) { 
        fscanf(file1, "%d", &A[i]);
        fscanf(file2, "%d", &B[i]);
    }

    fclose(file1);
    fclose(file2);

    pthread_t threads[num_of_threads];
    int thread_counter = 0;
    struct timeval start_time, end_time;
    gettimeofday(&start_time,NULL);

    for(int i=0; i<array_size; i += indices_per_thread) {
        int *index = (int*)malloc(sizeof(int));
        *index = i;
        pthread_create(&threads[thread_counter], NULL, addArrays, (void*)index);
        thread_counter++;
    }

    pthread_mutex_lock(&mutex);
    while(threads_completed != num_of_threads){
        pthread_cond_wait(&cond, &mutex);
    }
    pthread_mutex_unlock(&mutex);

    int sum = 0;
    for(int i=0; i<array_size; i++){
        sum += C[i];
    }
    gettimeofday(&end_time,NULL);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000.0 +
                          (end_time.tv_usec - start_time.tv_usec);

    printf("\nsum of elements:%d\n", sum);
    for(int i=0; i<num_of_threads; i++){
        pthread_join(threads[i], NULL);
    }
    
    printf("Time spent: %f us\n",elapsed_time);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    return 0;
}
