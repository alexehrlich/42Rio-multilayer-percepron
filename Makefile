all: setup

setup:
	@echo "Installing the dependencies"
	@python3 -m virtualenv venv
	@. venv/bin/activate && pip install -r ./requirements.txt &&

canc_preprocess_data:
	@. venv/bin/activate && \
	python3 ./cancer/data_preprocess.py

canc_train_classifier:
	@. venv/bin/activate && \
	export PYTHONPATH="$(PWD)" && \
	python3 ./cancer/train_canc_classifier.py

canc_predict_probe:
	@. venv/bin/activate && \
	export PYTHONPATH="$(PWD)" && \
	python3 ./cancer/prediction.py

mnist_train_classifier:
	@. venv/bin/activate && \
	export PYTHONPATH="$(PWD)" && \
	python3 ./mnist/train_mnist_classifier.py

mnist_predict_digit:
	@. venv/bin/activate && \
	export PYTHONPATH="$(PWD)" && \
	python3 ./mnist/classify_digit.py

clean:
	@rm -rf ./cancer/csv/created

fclean: clean
	@rm -rf ./venv
	@rm -rf ./cancer/plots
	@rm ./cancer/model.pkl
	@rm ./mnist/mnnist.pkl
	@rm -rf __pycache__
