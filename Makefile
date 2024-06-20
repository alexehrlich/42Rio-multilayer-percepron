ll: setup

setup:
	@echo "Installing the dependencies"
	@python3 -m venv venv
	@. venv/bin/activate && pip install -r ./requirements.txt

read_data:
	@. venv/bin/activate && python3 data_analyse.py

clean:
	rm -rf ./venv