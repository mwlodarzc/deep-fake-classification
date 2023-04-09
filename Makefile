.DEFAULT_GOAL: run
.PHONY: run clean

setup: requirements.txt
	pip install --upgrade pip
	pip install -r requirements.txt

run: setup
	python3 face_auth.py

clean:
	rm -rf __pycache__