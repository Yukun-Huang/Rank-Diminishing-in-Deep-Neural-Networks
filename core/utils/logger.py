import os
import os.path as osp
import sys


class Logger:
    def __init__(self, filename):
        os.makedirs(osp.split(filename)[0], exist_ok=True)
        self.name = filename
        self.file = open(filename, "w+", encoding='utf-8')
        self.alive = True
        self.stdout = sys.stdout
        sys.stdout = self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self.alive:
            sys.stdout = self.stdout
            self.file.close()
            self.alive = False

    def write(self, data):
        data += '\n'
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()
