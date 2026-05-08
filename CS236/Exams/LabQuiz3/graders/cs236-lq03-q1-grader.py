#!/usr/bin/env python3

# README
# the directory structure should be
# .
# ├── <rollnum>
# │   └── q1.patch
# ├── tcs
# │   └── {p1a.c,p1b.c,...}
# └── xv6-riscv-q1 (as given in the exam)
#     ├── q1-grader.py
#     └── {other xv6 files}
#
# from the xv6-riscv-q1/ directory run:
# chmod +x q1-grader.py
# ./q1-grader.py

import argparse, os, inspect, re, signal, subprocess, sys, time
from subprocess import run

parser = argparse.ArgumentParser()
parser.add_argument("rollno", help="roll no.")
args = parser.parse_args()


class QEMU(object):

    def __init__(self, reset=False):
        if reset:
            self.build_xv6()
            self.reset_fs()
        q = ["make", "qemu"]
        self.proc = subprocess.Popen(
            q, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        self.output = ""
        self.outbytes = bytearray()
        time.sleep(1)

    def reset_fs(self):
        try:
            run(["rm", "fs.img"], check=True)
            run(["make", "fs.img"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")

    def build_xv6(self):
        try:
            run(["make", "kernel/kernel"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")

    def save_output(self):
        try:
            with open("test-xv6.out", "w") as f:
                f.write(self.out)
                f.close()
        except OSError as e:
            print("Provided a bad results path. Error:", e)

    def cmd(self, c):
        if isinstance(c, str):
            c = c.encode("utf-8")
        self.proc.stdin.write(c)
        self.proc.stdin.flush()

    def crash(self):
        ps = run(
            ["ps", "-opid", "--no-headers", "--ppid", str(self.proc.pid)],
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
        kids = [int(line) for line in ps.stdout.splitlines()]
        if len(kids) == 0:
            print("no qemu")
            os.exit(1)
        # print("kill", kids[0])
        os.kill(kids[0], signal.SIGKILL)

    def stop(self):
        self.proc.terminate()

    def read(self):
        buf = os.read(self.proc.stdout.fileno(), 4096)
        self.outbytes.extend(buf)
        self.output = self.outbytes.decode("utf-8", "replace")
        print(self.output)

    def print_output(self):
        print(self.output)

    def lines(self):
        return self.output.splitlines()

    def error(self):
        print("FAIL: match failed", regexps)
        self.save_output()
        self.stop()
        sys.exit(1)

    def match(self, *regexps, exit=True):
        lines = self.lines()
        last = -1
        for i, line in enumerate(lines):
            if any(re.match(r, line) for r in regexps):
                print(line)
                last = i
        if last == -1 and exit:
            self.error()
        l = ""
        if last >= 0:
            l = lines[last]
        return last >= 0, l
    
    def chk_match(self, output_string):
        lines = self.output
        if output_string.strip() in lines.strip():
            return True
        return False

    def monitor(self, *regexps, progress="", timeout):
        deadline = time.time() + timeout
        while True:
            time.sleep(1)
            timeleft = deadline - time.time()
            if timeleft < 0:
                self.error()
            self.read()
            ok, _ = self.match(*regexps, exit=False)
            if ok:
                return
            ok, line = self.match(progress, exit=False)
            if ok:
                print(line)


outputs = [
'''mmap successful
munmap successful
''',
"""mmap successful
sbrk successful
munmap successful
Writing into unmapped area...
usertrap(): unexpected scause 0xf
""",
"""mmap successful
sbrk unsuccessful
munmap successful 
""",
"""mmap successful
sbrk unsuccessful
munmap successful
""",
"""mmap successful
munmap successful
sbrk successful
""",
"""mmap successful
mmap successful
sbrk unsuccessful
munmap successful
munmap successful
sbrk successful
""",
"""mmap unsuccessful
munmap unsuccessful
""",
"""mmap successfull""",
"""mmap successful
munmap successful
munmap unsuccessful
"""

]




def p1a():
    q = QEMU()
    q.cmd("p1a\n")
    time.sleep(1)
    q.read()
    
    ok = q.chk_match(outputs[0])
    q.crash()
    q.stop()
    return ok

def p1b():
    q = QEMU()
    q.cmd("p1b\n")
    time.sleep(1)
    q.read()
    ok = q.chk_match(outputs[1])
    q.crash()
    q.stop()
    return ok

def p1c():
    q = QEMU()
    q.cmd("p1c\n")
    time.sleep(1)
    q.read()
    ok = q.chk_match(outputs[2])
    q.crash()
    q.stop()
    return ok
    
def p1d():
    q = QEMU()
    q.cmd("p1d\n")
    time.sleep(1)
    q.read()
    ok = q.chk_match(outputs[3])
    q.crash()
    q.stop()
    return ok
    
def p1e():
    q = QEMU()
    q.cmd("p1e\n")
    time.sleep(1)
    q.read()
    ok = q.chk_match(outputs[4])
    q.crash()
    q.stop()
    return ok

def p1f():
    q = QEMU()
    q.cmd("p1f\n")
    time.sleep(1)
    q.read()
    ok = q.chk_match(outputs[5])
    q.crash()
    q.stop()
    return ok

def p1g():
    q = QEMU()
    q.cmd("p1g\n")
    time.sleep(1)
    q.read()
    ok = q.chk_match(outputs[6])
    q.crash()
    q.stop()
    return ok

def p1h():
    q = QEMU()
    q.cmd("p1h\n")
    time.sleep(1)
    q.read()
    ok = q.chk_match(outputs[7])
    q.crash()
    q.stop()
    return ok

def p1i():
    q = QEMU()
    q.cmd("p1i\n")
    time.sleep(1)
    q.read()
    ok = q.chk_match(outputs[8])
    q.crash()
    q.stop()
    return ok






def test_q1(rollno):
    for f in os.listdir(".."):
        if re.match(f".*{rollno}.*", f):
            break
    else:
        print("Not found")
        return
    subprocess.run(["git", "checkout", "."], check=True)
    subprocess.run(
        ["git", "apply", "--whitespace=fix" ,"--exclude=user/p1a.c" ,"--exclude=user/p1b.c" ,"--exclude=user/p1c.c" ,"--exclude=user/p1d.c" ,"--exclude=user/p1e.c" ,"--exclude=user/p1f.c" ,"--exclude=user/p1g.c" ,"--exclude=user/p1h.c" ,"--exclude=user/p1i.c", f"../{f}/q1.patch"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["cp", "../tcs/p1a.c" , "../tcs/p1b.c" , "../tcs/p1c.c" , "../tcs/p1d.c" , "../tcs/p1e.c" , "../tcs/p1f.c" , "../tcs/p1g.c" , "../tcs/p1h.c" , "../tcs/p1i.c" , "user/."], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["cp", "../Makefile" , "."], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    time.sleep(1)
    subprocess.run(["make", "clean"], check=True)
    proc = subprocess.Popen(
        ["make","-j8", "qemu"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    time.sleep(3)
    proc.terminate()
    marks = 0
    if proc.wait() == -15:
        if p1a():
            marks += 1
            print(marks)
            marks += 1 if p1b() else 0
            print(marks)
            marks += 1 if p1c() else 0
            print(marks)
            marks += 1 if p1d() else 0
            print(marks)
            marks += 1 if p1e() else 0
            print(marks)
            marks += 2 if p1f() else 0
            print(marks)
            marks += 1 if p1g() else 0
            print(marks)
            marks += 1 if p1h() else 0
            print(marks)
            marks += 1 if p1i() else 0
            print(marks)
    print(f"Marks: {marks}/10.0")
    with open("../final_marks.csv", "a") as f:
        f.write(f"{rollno},{marks}\n")
    subprocess.run(
        ["make", "clean"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["git", "checkout", "."], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )


def main():

    test_q1(args.rollno)


main()
