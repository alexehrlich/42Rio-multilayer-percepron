all: setup

setup:
	@echo "Installing the dependencies"
	@python3 -m venv venv
	@. venv/bin/activate && pip install -r ./requirements.txt

preprocess_data:
	@. venv/bin/activate && python3 data_preprocess.py

train:
	@. venv/bin/activate && python3 train.py

train_new:
	@. venv/bin/activate && python3 train_new.py

prediction:
	@. venv/bin/activate && python3 prediction.py

clean:
	rm -rf ./csv/created

fclean: clean
	rm -rf ./venv
