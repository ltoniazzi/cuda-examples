## Registers debug:

```
nvcc -arch=sm_80 flash_attention_spilling.cu -c -o flash_attention_spilling.o -Xptxas -v

# Get ptx
nvcc -arch=sm_80 -ptx flash_attention.cu -o flash_attention.ptx
```


ncu
```
nvcc -arch=sm_80 -O3 -lineinfo -o flash_attention flash_attention.cu

ncu --set full ./flash_attention


ncu --set full --csv --export profile.ncu-rep ./flash_attention

```