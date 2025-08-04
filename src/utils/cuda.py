import torch
import re
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline

def get_sig(fname, src):
    res = re.findall(rf'^(.+\s+{fname}\(.*?\))\s*{{?\s*$', src, re.MULTILINE)
    return res[0]+';' if res else None


def load_cuda(cuda_src, cpp_src, funcs, opt=True, verbose=False, name=None):
    "Simple wrapper for torch.utils.cpp_extension.load_inline"
    if name is None: name = funcs[0]
    flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if opt else "-O0 -Xptxas -O0 -Xcompiler -O0"
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=[flags], verbose=verbose, name=name)

def cdiv(a,b):
    "Int ceiling division of `a` over `b`"
    return (a+b-1)//b


if __name__ =="__main__":
    from pathlib import Path
    cuda_source_path = Path(__file__).parent.parent / "softmax/softmax.cu"
    cuda_source = cuda_source_path.read_text()
    fname = "softmax"
    cpp_source = get_sig(fname, cuda_source)
    # print(cpp_source)
    module = load_cuda(cuda_source, cpp_source, funcs=[fname], verbose=True, opt=True)