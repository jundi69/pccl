import torch

if __name__ == '__main__':
    a = torch.load("weight_dumpcuda:0.pt").to('cpu')
    b = torch.load("weight_dumpcuda:1.pt").to('cpu')

    print(torch.all(a.view(dtype=torch.uint32) == b.view(torch.uint32)))