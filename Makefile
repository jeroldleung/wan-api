.PHONY: install lint

install:
		git lfs install
		git clone https://www.modelscope.cn/Wan-AI/Wan2.1-T2V-1.3B.git pretrained_models/Wan2.1-T2V-1.3B

lint:
		uv run ruff format
		uv run ruff check --exclude inference.ipynb --fix
		