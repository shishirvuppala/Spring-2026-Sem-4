PARTS := p2a p2b p2c p2d p2e

all: $(PARTS)

%: %.c
	gcc -o $@ $<

clean: 
	rm -f $(PARTS)
