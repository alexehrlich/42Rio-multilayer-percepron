all: setup

setup:
	@echo "Installing the dependencies"
	@python3 -m venv venv
	@. venv/bin/activate && pip install -r ./requirements.txt

preprocess_data:
	@. venv/bin/activate && python3 data_analyse.py

train:
	@. venv/bin/activate && python3 train.py

prediction:
	@. venv/bin/activate && python3 prediction.py

clean:
	rm target_test.csv
	rm target_train.csv
	rm features_test.csv
	rm features_train.csv

fclean: clean
	rm -rf ./venv
