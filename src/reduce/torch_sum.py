import torch


device = "cpu"

x = torch.Tensor([1, 2, 3]).to("cuda")

res = torch.sum(x)

print(res)


def launch_reduction(input_array, reduction, init):
    output = init
    for element in input_array:
        output = reduction(output, element) 

    return output


sum = lambda a, b: a + b

input_array = [1, 2, 3]

res = launch_reduction(
    input_array=input_array,
    reduction=sum,
    init=0,
)
print(res)





input = [1, 2, 3]
output = None

def sum_kernel(
    input: list[float], 
    output: float,
    input_size: int
    ):
    sum = 0
    for pos in range(input_size):
        sum += input[pos]

    output = sum






    return output





