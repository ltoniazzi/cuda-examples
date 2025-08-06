

setup:
	virtualenv --python=python3 .venv
	.venv/bin/pip install -r requirements.txt
	sudo apt-get install ninja-build


rm_cache:
	rm -rf ~/.cache/torch_extensions/py312_cu126