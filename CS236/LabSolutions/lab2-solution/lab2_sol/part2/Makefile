PARTS := p2a p2b p2c p2d

all: $(PARTS)

%: %.c
	gcc -o $@ $<

clean: 
	rm -f $(PARTS)
	find . -maxdepth 1 -type f -name '*.txt' -and -not -name 'input.txt' -delete
