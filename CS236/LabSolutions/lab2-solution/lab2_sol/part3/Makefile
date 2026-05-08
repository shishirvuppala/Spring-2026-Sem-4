PARTS := p3a p3b p3c p3d

all: $(PARTS)

%: %.c
	gcc -o $@ $<

clean: 
	rm -f $(PARTS)
