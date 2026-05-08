PARTS := p5a p5b

all: $(PARTS)

%: %.c
	gcc -o $@ $<

clean: 
	rm -f $(PARTS)
	find . -maxdepth 1 -type f -name '*.txt' -and -not -name 'dummy_file.txt' -delete
