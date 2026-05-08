#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define BUFFER_SIZE 100

int main(int argc, char* argv[]) {
    char buf[BUFFER_SIZE + 1];
    int fd;
    long offset;

    if (argc != 3) {
        printf("Usage: %s <filename> <offset>\n", argv[0]);
        return 1;
    }

    offset = atoi(argv[2]);
    fd = open(argv[1], O_RDONLY);

    if (fd == -1) {
        perror("Error opening");
        return 1;
    }

    int cid = fork();
    if (cid == 0) {
        sleep(5);

        int n = read(fd, buf, BUFFER_SIZE);
        if (n < 0) {
            perror("Error reading");
            close(fd);
            exit(1);
        }

        buf[n] = '\0';
        printf("Child read: %s\n", buf);

        close(fd);
    } else {
        lseek(fd, offset, SEEK_SET);

        int n = read(fd, buf, BUFFER_SIZE);
        if (n < 0) {
            perror("Error reading");
            close(fd);
            return 1;
        }

        buf[n] = '\0';
        printf("Parent read: %s\n", buf);

        wait(NULL);
        close(fd);
    }

    return 0;
}
