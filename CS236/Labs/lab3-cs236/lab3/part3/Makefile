PARTS := p3a shell

all: $(PARTS)

%: %.c
	gcc -o $@ $<

clean: 
	rm -f $(PARTS)
