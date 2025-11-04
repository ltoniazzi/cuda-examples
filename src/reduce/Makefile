
FILE_NAME ?= trivial_reduce


run:
	nvcc -o $(FILE_NAME) $(FILE_NAME).cu && ./$(FILE_NAME) 

run_trivial:
	$(MAKE) run FILE_NAME=trivial_reduce

run_thread:
	$(MAKE) run FILE_NAME=thread_reduce

run_block:
	$(MAKE) run FILE_NAME=block_reduce

run_block_unroll:
	$(MAKE) run FILE_NAME=block_reduce_unroll

run_all:
	$(MAKE) run_trivial
	$(MAKE) run_thread
	$(MAKE) run_block


prof:
	ncu --set full $(FILE_NAME)

prof_block:
	$(MAKE) run_block
	$(MAKE) prof FILE_NAME=block_reduce

prof_trivial:
	$(MAKE) run_trivial
	$(MAKE) prof FILE_NAME=trivial_reduce | grep msec
	
prof_thread:
	$(MAKE) run_thread
	$(MAKE) prof FILE_NAME=thread_reduce

prof_block_and_torch:
	$(MAKE) run_block
	@echo "Our implementation no unroll"
	$(MAKE) prof FILE_NAME=block_reduce | grep msec
	@echo "Our implementation with unroll"
	$(MAKE) prof FILE_NAME=block_reduce_unroll | grep msec
	@echo "Torch CPU and GPU implementations"
	python ./benchmark_torch_sum.py