import time
import subprocess

from pccl import *

HOST: str = '0.0.0.0:48148'


def main():
    pass


if __name__ == '__main__':
    print(f"Starting master node on {HOST}")
    master: MasterNode = MasterNode(listen_address=HOST)
    master.run()
