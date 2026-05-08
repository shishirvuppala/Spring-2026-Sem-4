#!/bin/bash

git -C xv6-riscv/ add -A
git -C xv6-riscv/ diff HEAD >lab5.patch
