import os
import sys

print(f'cwd: {os.getcwd()}')
#print(f'pythonpath: {os.environ["PYTHONPATH"]}')
try:
    print(f'\nenvirons: {os.environ["ASDF"]}')
except Exception as e:
    print(f'exception: {e}, {type(e)}')
    
print(f'\nsys.path: {sys.path}')