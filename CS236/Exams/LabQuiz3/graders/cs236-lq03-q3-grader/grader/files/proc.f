int testf(char *fmt, ...);

int
read_runtime(void){
  int pid;
  argint(0, &pid);
  struct proc *p;
  for(p = proc; p < &proc[NPROC]; p++){
    if(p->state == UNUSED)
      continue;

    if (p->pid == pid){
      return p->virtual_runtime;
    }
  }
  return -1;
}

void
procdump(void)
{
  static char *states[] = {
  [UNUSED]    "unused",
  [USED]      "used",
  [SLEEPING]  "sleep ",
  [RUNNABLE]  "runble",
  [RUNNING]   "run   ",
  [ZOMBIE]    "zombie"
  };
  struct proc *p;
  char *state;

  printf("\n");
  printf("=====UNFAIR STATE OF VIRTUAL RUNTIMES=====\n");
  printf("PID STATE NAME VIRTUAL_RUNTIME\n");
  printf("\n");

  for(p = proc; p < &proc[NPROC]; p++){
    if(p->state == UNUSED)
      continue;
    if(p->state >= 0 && p->state < NELEM(states) && states[p->state])
      state = states[p->state];
    else
      state = "???";
    printf("%d %s %s %d", p->pid, state, p->name, p->virtual_runtime);
    printf("\n");
  }
}

int schedulingCheck(){
  struct proc *p, *pp;
  int maxticks;
  int display;
  int *output;

  argint(0, &display);
  argint(1, &maxticks);
  argaddr(2, (uint64*) &output);

  int result[3] = {0, 0, 0};

  struct proc *myp = myproc();

  uint64 baseTick = ticks;

  static char *states[] = {
  [UNUSED]    "unused",
  [USED]      "used",
  [SLEEPING]  "sleep ",
  [RUNNABLE]  "runble",
  [RUNNING]   "run   ",
  [ZOMBIE]    "zombie"
  };

  while(ticks - baseTick < maxticks){
    for(p = proc; p < &proc[NPROC]; p++){
      for(pp = proc; pp < &proc[NPROC]; pp++){
        if(p->state == UNUSED || pp->state == UNUSED)
          continue;
        if (p->state == RUNNABLE && pp->state == RUNNABLE){
          if (pp->virtual_runtime - p->virtual_runtime > TIMESLICE || 
              p->virtual_runtime - pp->virtual_runtime > TIMESLICE){
            result[0]++;
            if (!display || result[0] > 10)
              continue;
            printf("\tUNFAIR STATE\n");
            printf("%d %s %s %d\n", p->pid, states[p->state], p->name, p->virtual_runtime);
            printf("%d %s %s %d\n", pp->pid, states[pp->state], pp->name, pp->virtual_runtime);
            printf("\n");
          }
        }
        result[1]++;
      }
    }
    result[2]++;
  }
  copyout(myp->pagetable, (uint64) output, (char *)result,
                                  sizeof(result));
  return 0;
}

int
waitnohang(uint64 addr)
{
  struct proc *pp;
  int havekids, pid;
  struct proc *p = myproc();

  acquire(&wait_lock);

  // Scan through table looking for exited children.
  havekids = 0;
  for(pp = proc; pp < &proc[NPROC]; pp++){
    if(pp->parent == p){
      // make sure the child isn't still in exit() or swtch().
      acquire(&pp->lock);

      havekids = 1;
      if(pp->state == ZOMBIE){
        // Found one.
        pid = pp->pid;
        if(addr != 0 && copyout(p->pagetable, addr, (char *)&pp->xstate,
                                sizeof(pp->xstate)) < 0) {
          release(&pp->lock);
          release(&wait_lock);
          return -1;
        }
        freeproc(pp);
        release(&pp->lock);
        release(&wait_lock);
        return pid;
      }
      release(&pp->lock);
    }
  }

  // No point waiting if we don't have any children.
  if(!havekids || killed(p)){
    release(&wait_lock);
    return -1;
  }
  
  // Wait for a child to exit.
  release(&wait_lock);
  return -2;
}


#define MAXTICKS 200
const int DISPLAY_THRESHOLD = 2; 

int policy_errors[NCPU] = {0};
int policy_checks[NCPU] = {0};
int scheduled_ticks[NPROC][MAXTICKS] = {0};
int vruntimes[NPROC][MAXTICKS] = {0};
int minimum_vruntimes[NPROC][MAXTICKS] = {0};
int num_ticks[NPROC] = {0};

void logger(uint64 from, uint64 to){
  push_off();
  struct proc *myp= myproc();
  int pcb_index = myp - proc;
  
  if (num_ticks[pcb_index] < MAXTICKS){
    scheduled_ticks[pcb_index][num_ticks[pcb_index]] = ticks;
    vruntimes[pcb_index][num_ticks[pcb_index]] = myp->virtual_runtime;
    num_ticks[pcb_index]++;
  }
  
  if (((uint64) &myp->context) == to){
    struct proc *p;
    int minimum_vruntime = myp->virtual_runtime;
    for(p = proc; p < &proc[NPROC]; p++){
      if (p->state == RUNNABLE && p->virtual_runtime < minimum_vruntime){
        minimum_vruntime = p->virtual_runtime;
      }
    }
    minimum_vruntimes[pcb_index][num_ticks[pcb_index] >> 1] = minimum_vruntime;
  }
  pop_off();
}


int myticks(uint64 addr){
  return copyout(myproc()->pagetable, addr, (char*)scheduled_ticks[myproc() - proc], MAXTICKS * sizeof(int));
}

int myvruntimes(uint64 addr){
  return copyout(myproc()->pagetable, addr, (char*)vruntimes[myproc() - proc], MAXTICKS * sizeof(int));
}

int myminimumvruntimes(uint64 addr){
  return copyout(myproc()->pagetable, addr, (char*)minimum_vruntimes[myproc() - proc], MAXTICKS * sizeof(int));
}

int get_policy_result(uint64 addr){
  int result[2] = {0};
  for(int i = 0; i < NCPU; i++){
    result[0] += policy_errors[i];
    result[1] += policy_checks[i];
  }

  return copyout(myproc()->pagetable, addr, (char *)result,
                                sizeof(result) * sizeof(int));
}