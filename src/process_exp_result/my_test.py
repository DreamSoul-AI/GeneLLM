# run this file at /src: python python_test/my_test.py
import sys
import os
# add src to sys.path, so that we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from module import load

print(load)