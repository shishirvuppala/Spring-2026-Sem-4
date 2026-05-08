#include "kernel/types.h"
#include "kernel/stat.h"
#include "kernel/fcntl.h"
#include "user/user.h"

int main(int argc, char *argv[])
{  
    if(argc != 4){
        fprintf(2, "Usage: %s <file> <mode> <perm>\n", argv[0]);
        exit(1);
    }
    int mode = -1;
    int perm = atoi(argv[3]);
    if(argv[2][0] == 'r' ){
        mode =0;
    }
    if(argv[2][0] == 'w' ){
        mode =1;
    }

    if( setPriv(argv[1], mode, perm))
        printf("set priv success\n");
    else
        printf("set priv failed\n");
}