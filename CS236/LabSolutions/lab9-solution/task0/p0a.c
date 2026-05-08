#include <stdio.h>

int total = 0;

void increment(int x) {
    for (int i = 0; i < 1000000; i++) {
        total += x;
    }
}

int main() {
    for (int i = 0; i < 4; i++) {
        increment(i + 1);
    }
    printf("Total: %d\n", total);
    return 0;
}