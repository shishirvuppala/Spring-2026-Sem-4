PARTS := p2a p2b

all: $(PARTS)

%: %.c
	gcc -o $@ $<

clean: 
	rm -f $(PARTS)