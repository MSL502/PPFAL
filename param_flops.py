from thop import profile
from models.network import Network as Net
import torch
import time
if __name__ == '__main__':
    net = Net('test')
    net = net.to(torch.device('cuda:1'))
    input = torch.randn(1, 3, 256, 256)
    input = input.to(torch.device('cuda:1'))
    flops, param = profile(net, (input,))
    print(f'flops: {flops/1e9} G\n')
    print(f'param: {param/1e6} MB')