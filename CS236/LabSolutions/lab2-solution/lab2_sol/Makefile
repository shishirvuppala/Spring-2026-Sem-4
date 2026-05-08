QS := part1 part2 part3 part4 part5
 
.PHONY: $(QS)

all: $(QS)

$(QS):
	$(MAKE) -C $@

clean:
	for d in $(QS); do \
		$(MAKE) -C $$d clean; \
	done
