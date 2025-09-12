import subprocess
import sys
from time import sleep

c = open('./o.io','w')
c.write(sys.argv[1])
c.close()

#subprocess.run('notepad.exe ./t.txt