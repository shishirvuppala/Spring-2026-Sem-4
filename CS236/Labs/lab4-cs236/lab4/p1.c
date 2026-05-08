#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int main(int argc,char* argv[])
{
 if(argc != 2){
	printf("Usage: p1 <filename>\n");
} else{
	printf("%s\n",argv[1]);
}

 // Write your code here
 
  exit(0);
}


