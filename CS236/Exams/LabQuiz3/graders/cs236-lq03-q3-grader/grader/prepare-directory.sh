#!/bin/bash
cd submission-in-test
git add .
git reset --hard HEAD
git apply --whitespace=fix ../$1
        
# Replace files
cp ../files/swtch.S kernel/swtch.S
cp ../files/printf.c kernel/printf.c  
cp ../files/syscall.c kernel/syscall.c
cp ../files/syscall.h kernel/syscall.h
cp ../files/user.h user/user.h
cp ../files/usys.pl user/usys.pl
cp ../files/tc-slice.c user/tc-slice.c
cp ../files/tc-vruntime.c user/tc-vruntime.c
cp ../files/tc-policy.c user/tc-policy.c

# Add files
cp ../files/testing.h kernel/testing.h
cp ../files/tc-fork.c user/tc-fork.c
cp ../files/tc-wakeup.c user/tc-wakeup.c


# Prepare sysproc.c
sed -i '1i #include "testing.h"' kernel/sysproc.c
echo -e '\n#include "../files/sysproc.f"' >> kernel/sysproc.c

# Prepare proc.c
sed -i '1i #include "testing.h"' kernel/proc.c
echo -e '\n#include "../files/proc.f"' >> kernel/proc.c