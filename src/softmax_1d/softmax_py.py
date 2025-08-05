from collections import namedtuple
import torch
dim3 = namedtuple('dim3', ['x','y','z'], defaults=(1,1))

def softmax_py_kernel(blockId: dim3, blockDim: dim3, threadId, V: torch.Tensor, O: torch.Tensor, s: int):
    """
    Compute softmax of V using O as the output tensor.
    """
    res = 0.0
    tot = 0.0
    location = blockId.x*blockDim.x + threadId.x

    if location >= s:
        return

    for i in range(s):
        cur = V[i] # s reads * s threads
        cur = torch.exp(cur) # s exp * s threads
        tot += cur # s sum * s threads
        if i == location:
            res = cur
        

    res = res/tot  # 1 div * s threads

    O[location] = res # 1 write * s threads


def cdiv(a, b):
    return (a + b - 1) // b

def blk_kernel1d(f, blocks, threads, *args):
        for i1 in range(blocks.x):
                for j1 in range(threads.x): 
                    f(dim3(i1), dim3(blocks.x), dim3(j1), *args)
     
def softmax_py(V): 
    s = V.shape[0]
    tpb = 32  # threads per block
    threads = dim3(tpb, 1, 1)
    blocks = dim3(cdiv(s, tpb), 1, 1)
    O = torch.zeros_like(V, dtype=torch.float32)

    blk_kernel1d(
         softmax_py_kernel, blocks, threads, V, O, s
    )

    return O


if __name__ == "__main__":
    V = torch.randn(32, dtype=torch.float32)

    O = softmax_py(V)
    # print("Softmax output:", O)

    O_torch = torch.softmax(V, dim=0)
    print(torch.allclose(O, O_torch, atol=1e-4))
