#include <stdio.h>
#include <unistd.h>

int main() {
    // TODO: fork and complete if-else (copy from p3b.c)
    ;
    if () {
        printf("fork failed\n");
        return 1;
    } else if () {  // Child
        // TODO: print (copy from p3b.c)
        ;

        // TODO: sleep for 5 secs using sleep syscall
        ;
    } else {  // Parent
        // TODO: wait for user input before waiting for child
        ;

        // TODO: wait for child to terminate using wait syscall
        ;

        // TODO: print (copy from p3b.c)
    }

    return 0;
}
