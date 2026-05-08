#!/bin/bash

CONTAINER_NAME="xv6-devbox"

if [ "$(docker ps -q -f name=xv6-devbox)" ]; then
  docker exec -it "$CONTAINER_NAME" bash
else
  docker run --rm -it \
    --name "$CONTAINER_NAME" \
    -h xv6 \
    -u $(id -u):$(id -g) \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v ./xv6-riscv:/xv6-riscv \
    --network host \
    arghyadipchak/xv6-riscv-devbox:cs236-lab12
fi
