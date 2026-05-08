PARTS := p3a p3b

all: $(PARTS)

%: %.c
	gcc -o $@ $<

clean: 
	rm -f $(PARTS)