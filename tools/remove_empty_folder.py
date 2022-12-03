import os

walk = list(os.walk(os.getcwd()))
for path, _, _ in walk[::-1]:
    if len(os.listdir(path)) == 0:
        os.rmdir(path)