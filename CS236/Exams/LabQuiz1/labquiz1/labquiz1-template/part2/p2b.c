#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

// Use the provided print statements in your logic.
// Declare any global variables you need.

int N;

// void print_info(int process_type, int pid , int count)
// {
//     if (process_type == 0)
//         printf("[pid %d]: Parent Received Signal %d\n", ___, ___);

//     else
//         printf("[pid %d]: Child Received Signal %d\n", ___, ___);
// }

// TODO:
// ___ handler_1(___) {
//     TODO: some other actions
// }
//
// ___ handler_2(___) {
//     TODO: some other actions
// }
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);

    /* ---------------- WRITE YOUR CODE HERE ---------------- */

    /* Use the following print statements in your logic:
     * printf("[pid %d]: Parent Process\n", ___);
     * printf("[pid %d]: Child Process\n", ___);
     *
     * parent and child process may want to use some different paths.
     * if (___) {
     *
     *
     * } else {
     *
     * }
     */

    return 0;
}
