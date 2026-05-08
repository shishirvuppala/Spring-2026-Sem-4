#!/bin/bash

docker run --rm -it \
  --name xv6-devbox \
  -h xv6 \
  -u $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v ./xv6-riscv:/xv6-riscv \
  arghyadipchak/xv6-riscv-devbox:latest
