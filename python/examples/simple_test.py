from pccl import *

m = MasterNode(listen_address='127.0.0.1:8080')
m.run()

c = Communicator()
c.connect('127.0.0.1:8080')

m.interrupt()