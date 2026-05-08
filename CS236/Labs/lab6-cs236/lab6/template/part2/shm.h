#ifndef __SHM_H__
#define __SHM_H__

#include "types.h"

struct shmseg {
    int valid;
    int refcnt;
    uint64 kva;
};

static struct shmseg shm;

#endif
