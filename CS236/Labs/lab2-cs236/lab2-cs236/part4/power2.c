#include <math.h>
#include <stdio.h>
#include <unistd.h>

int power(int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; i++) result *= base;

    return result;
}

int main() {
    int n;

    printf("Enter a positive integer: ");
    scanf("%d", &n);
    if (n < 0) {
        printf("Please enter a positive integer!\n");
        return 1;
    }

    int result = power(2, n);
    printf("2^%d = %d\n", n, result);

    char* argv[] = {"power3", NULL};
    execv(argv[0], argv);

    printf("Exiting from power2.c\n");

    return 0;
}