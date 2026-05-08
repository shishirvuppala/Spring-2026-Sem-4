import os
from subprocess import run
import subprocess
import sys
import time
import select
import re
import fcntl

class colors:
    HEADER = '\033[94m'
    OKGREEN = '\033[92m'
    LOG = '\033[93m'
    FAIL = '\033[95m'
    END = '\033[0m'
    TOTAL = '\033[96m'
    ERROR = '\033[91;1m' 


output_file = open("result.csv", "w")
log_file = open("result.log", "w")
output_file.write("Total Marks, tc-slice, tc-vruntime, tc-fork, tc-wakeup, tc-policy-1, tc-policy-2\n")

def log(line: str):
    print(f"{colors.END}{line}{colors.END}", end = "")
    print(f"{line}", file=log_file, end = "")

def log_passed(line: str):
    print(f"{colors.OKGREEN}[VERDICT]:  {line} {colors.END}")
    print(f"[VERDICT]:  {line} ", file=log_file)

def log_failed(line: str):
    print(f"{colors.FAIL}[VERDICT]:  {line} {colors.END}")
    print(f"[VERDICT]:  {line} ", file=log_file)

def log_error(line: str):
    print(f"{colors.ERROR}[ERROR]:  {line} {colors.END}")
    print(f"[ERROR]:  {line} ", file=log_file)

def log_total_score(line: str):
    print(f"{colors.TOTAL}[TOTAL SCORE]:  {line} {colors.END}")
    print("=================================================\n")

    print(f"[TOTAL SCORE]:  {line} ", file=log_file)
    print("=================================================\n", file=log_file)

def log_flush():
    sys.stdout.flush()
    log_file.flush()


def log_header(line: str):
    print(f"{colors.HEADER}", end="")
    print("=================================================")
    print(f"\t\t{line}")
    print("=================================================")
    print(f"{colors.END}", end="")

    print(f"", end="", file=log_file)
    print("=================================================", file=log_file)
    print(f"\t\t{line}", file=log_file)
    print("=================================================", file=log_file)
    print(f"", end="", file=log_file)



def set_nonblocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

class QEMU(object):

    def __init__(self, reset=False):
        self.proc = None
        if reset:
            self.build_xv6()
            self.reset_fs()
        q = ["stdbuf", "-o0", "make", "qemu"]
        self.proc = subprocess.Popen(q, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, bufsize=0)
        set_nonblocking(self.proc.stdout.fileno())
        self.output = ""
        self.outbytes = bytearray()       
        self.wait_for_prompt(20)

    def reset_fs(self):
        try:
            run(["make", "fs.img"], check=True,stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            log_error(f"Reset_FS failed with exit code {e.returncode}")
            raise RuntimeError("Compilation unsuccessful")

    def build_xv6(self):
        try:
            run(["make", "kernel/kernel"], check=True,stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            log_error(f"Build failed with exit code {e.returncode}")
            raise RuntimeError("Compilation unsuccessful")  
        
    def wait_for_prompt(self, timeout=20):
        deadline = time.time() + timeout
        pattern = r"^\$"

        while time.time() < deadline:
            self.read()

            if re.search(pattern, self.output, re.MULTILINE):
                return

        raise RuntimeError("QEMU did not boot in time")
        
    def cmd(self, c):
        if isinstance(c, str):
            c = c.encode('utf-8')
        self.proc.stdin.write(c)
        self.proc.stdin.flush()

    def stop(self):
        if self.proc is not None:
            self.proc.stdin.write(b'\x01x') 
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
            self.proc = None

    def read(self, timeout = 0.2):
        prev_output = self.output
        deadline = time.time() + timeout
        while time.time() < deadline:
            r, _, _ = select.select([self.proc.stdout.fileno()], [], [], 
                                        max(0, deadline - time.time()))
            if not r:
                break
            chunk = os.read(self.proc.stdout.fileno(), 4096)
            if not chunk:
                break
            self.outbytes.extend(chunk)

        self.output = self.outbytes.decode("utf-8", "replace")
        if len(self.output) > len(prev_output):
            log(self.output[len(prev_output):])
            log_flush()

    def lines(self):
        return self.output.splitlines()

    def match(self, regexp):
        matched_result = re.search(regexp, self.output, re.MULTILINE)
        if matched_result is not None:
            return True, matched_result.group().splitlines()
        return False, []

    def monitor(self, good, bad, progress="", timeout=60):
        deadline = time.time() + timeout
        passed = False
        while time.time() < deadline:

            self.read(timeout = 0.2)
            
            passed, _ = self.match(good)
            failed, _ = self.match(bad)

            terminate, _ = self.match(r"^\$([\s\S]*)^\$")

            if terminate:
                if passed and not failed:
                    return 1
                elif failed and not passed:
                    return 0
                else:
                    assert False, "Output compromised"
        return 2

def test(testcase, marks, timeout):
    log_header(f"Running {testcase}")
    q = None
    try:
        q = QEMU(True)
    except Exception as e:
        log_error(e)
        log_failed(f"[0/{marks}] (Compilation failed)")
        if q is not None:
            q.stop()
        return 0, f"[0/{marks}] (Compilation failed)"
    try:

        q.cmd(testcase + "\n")
        result = q.monitor(good = f'\t{testcase} PASSED', bad = f'\t{testcase} FAILED', progress="", timeout=timeout)
        q.stop()
        log("\n")

        if result == 1:
            verdict = f"[{marks}/{marks}]"
            log_passed(verdict)
            return marks, verdict
        elif result == 2:
            verdict = f"[0/{marks}] (Timed out)"
            log_failed(verdict)
            return 0, verdict
        else:
            verdict = f"[0/{marks}] (Failed)"
            log_failed(verdict)
            return 0, verdict
        
    except Exception as e:
        log_error(e)
        if q is not None:
            q.stop()
        assert False, "Failed to monitor correctly"
    

def check():
    log_header(f"Grading")
    os.chdir('submission-in-test')

    testcases = ['tc-slice', 'tc-vruntime', 'tc-fork', 'tc-wakeup', 'tc-policy']
    timeouts = [120, 120, 120, 120, 200]
    marks = [2, 1, 1, 1, 5]
    verdict = ["" for _ in range(5)]
    scores = [0 for _ in range(5)]

    try:
        for i in range(5):
            if i in [0, 1, 2, 3]:
                subprocess.run(["cp", "../files/Makefile1", "Makefile"])
            elif i in [4]:
                subprocess.run(["cp", "../files/Makefile5", "Makefile"])
            
            subprocess.run(["make", "clean"], check=True,stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)

            scores[i], verdict[i] = test(testcases[i], marks[i], timeouts[i])
            # Stop if timeslice is not implemented correctly
            if i in [0] and not scores[0]:
                log_error("Timeslice not implemented correctly, skipping remaining tests")
                break
            # Stop if vruntime is not updated correctly
            if i in [3] and (not scores[1] or not scores[2] or not scores[3]):
                log_error("Vruntime not updated correctly, skipping remaining tests")
                break
            time.sleep(1)

    except Exception as e:
        log_error(e)
    
    log_header(f"Graded")
    log_total_score(f"[{sum(scores)}/10]")

    output_file.write(f"{sum(scores)},{",".join(verdict)}\n")



if __name__ == "__main__":
    check()
