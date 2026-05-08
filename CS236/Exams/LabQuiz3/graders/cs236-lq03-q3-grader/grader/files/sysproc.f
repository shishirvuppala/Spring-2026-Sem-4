int waitnohang(uint64);
int get_policy_result(uint64);
int read_runtime(void);
int myticks(uint64);
int myvruntimes(uint64);
int myminimumvruntimes(uint64);
int testf(char *fmt, ...);


uint64 
sys_read_runtime(void){
  return read_runtime();
}

uint64
sys_waitnohang(void){
  uint64 p;
  argaddr(0, &p);
  return waitnohang(p);
}

uint64
sys_get_policy_result(void){
  uint64 addr;
  argaddr(0, &addr);
  return get_policy_result(addr);
}

uint64
sys_myticks(void){
  uint64 addr;
  argaddr(0, &addr);
  return myticks(addr);
}

uint64
sys_myvruntimes(void){
  uint64 addr;
  argaddr(0, &addr);
  return myvruntimes(addr);
}

uint64
sys_myminimumvruntimes(void){
  uint64 addr;
  argaddr(0, &addr);
  return myminimumvruntimes(addr);
}
