#include <math.h>
#include <stdio.h>

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

    int result = power(3, n);
    printf("3^%d = %d\n", n, result);

    printf("Exiting from power3.c\n");

    return 0;
}
