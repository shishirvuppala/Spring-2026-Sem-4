PARTS := p3a p3b p3c

all: $(PARTS)

%: %.c
	gcc -o $@ $<

clean: 
	rm -f $(PARTS)
