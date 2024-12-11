import pccl

m = MasterNode(listen_address='127.0.0.1:8080')
m.run()

c = Communicator()
c.connect_master('127.0.0.1:8080')

m.interrupt()