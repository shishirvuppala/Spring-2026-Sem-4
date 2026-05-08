#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define BUFFER_SIZE 100

int main(int argc, char* argv[]) {
    char buf[BUFFER_SIZE + 1];
    int fd, n;
    long offset;

    if (argc != 3) {
        printf("Usage: %s <filename> <offset>\n", argv[0]);
        return 1;
    }

    offset = atoi(argv[2]);

    // TODO: open file fd for reading
    ;

    // TODO: fork a child process
    ;
    if () {        // Child
        sleep(5);  // wait for parent to read first

        // TODO: read from fd into buf
        ;

        buf[n] = '\0';
        printf("Child read: %s\n", buf);

        // TODO: close fd
        ;
    } else {  // Parent
        // TODO: seek to offset in fd using lseek syscall
        lseek();

        // TODO: read from fd into buf
        ;

        buf[n] = '\0';
        printf("Parent read: %s\n", buf);

        wait(NULL);
        // TODO: close fd
        ;
    }

    return 0;
}
