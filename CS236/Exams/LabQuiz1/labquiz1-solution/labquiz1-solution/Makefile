CC = gcc

all: send_siguser1 p1a p1b p2a p2b p3a p3b p3c

p1a: p1a.c
	$(CC) p1a.c -o p1a

p1b: p1b.c
	$(CC) p1b.c -o p1b

p2a: p2a.c
	$(CC) p2a.c -o p2a

p2b: p2b.c
	$(CC) p2b.c -o p2b

p3a: p3a.c
	$(CC) p3a.c -o p3a

p3b: p3b.c
	$(CC) p3b.c -o p3b

p3c: p3c.c
	$(CC) p3c.c -o p3c

send_siguser1: send_siguser1.c
	$(CC) send_siguser1.c -o send_siguser1

clean:
	rm -f send_siguser1 p1a p1b p2a p2b p3a p3b p3c