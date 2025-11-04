# Reduce/Sum

Simple implementations to grasp basic understanding of reduce operations on GPU. 

1. Build a trivial single thread kernel
2. Spread parallel reduction over multiple threads (see limit of num threads per block)
3. Spread parallel reduction over blocks (done in a simple manner)
4. Performance compare with CPU and GPU pytorch


Follows some ideas from the Reductions lecture of GPU Mode ([video](https://www.youtube.com/watch?v=09wntC6BT5o), [code](https://github.com/gpu-mode/lectures/tree/main/lecture_009))


## Main commands

```bash
make run_all
make prof_block
```