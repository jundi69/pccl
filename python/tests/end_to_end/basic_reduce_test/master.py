import time
import subprocess

from pccl import *

HOST: str = '0.0.0.0:48148'


def main():
    print(f"Starting master node on {HOST}")
    master: MasterNode = MasterNode(listen_address=HOST)
    master.run()

    time.sleep(1)
    print(f"Master node started.")

    p = subprocess.run(["lsof", "-i", "-P", "-n"], capture_output=True)
    print(p.stdout.decode())


if __name__ == '__main__':
    main()
    print("!!Master has exited!!")
