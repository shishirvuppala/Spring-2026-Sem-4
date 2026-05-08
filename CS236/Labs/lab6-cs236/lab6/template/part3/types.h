typedef enum {
  CMD_ECHO = 0,
  CMD_GETPID = 1,
  CMD_GETPPID = 2,
  CMD_UPTIME = 3,
  CMD_PROC_COUNT = 4,
} shm_cmd_t;

struct shm_channel {
  // User → Kernel
  volatile uint8 u_ready;
  volatile uint8 u_ack;
  volatile shm_cmd_t u_cmd;

  // Kernel → User
  volatile uint8 k_ready;
  volatile uint8 k_ack;
  volatile uint64 k_res;
};
