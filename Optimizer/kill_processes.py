import subprocess
import os
subprocess = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
output, error = subprocess.communicate()
print(output)
target_process = "python"
for line in output.splitlines():
    if target_process in str(line):
        pid = int(line.split(None, 1)[0])
        os.kill(pid, 9)