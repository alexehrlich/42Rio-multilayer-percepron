import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import os
import random

def train_val_test_split(data, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
	if (train_ratio+ validation_ratio + test_ratio != 1.0):
		print("Wrong split ratio. Must be equal to 1.0 in total.")
		return

	shuffeld_df = data.copy()
	shuffeld_df = shuffeld_df.sample(frac=1, random_state=42).reset_index(drop=True)

	data_len = len(data)
	train_len = int(data_len * train_ratio)
	validation_len = int(data_len * validation_ratio)

	Y_train = shuffeld_df.iloc[:train_len, 0]
	X_train = shuffeld_df.iloc[:train_len, 1:]

	Y_valid = shuffeld_df.iloc[train_len:train_len+validation_len, 0]
	X_valid = shuffeld_df.iloc[train_len:train_len+validation_len, 1:]

	Y_test = shuffeld_df.iloc[train_len+validation_len:, 0]
	X_test = shuffeld_df.iloc[train_len+validation_len:, 1:]

	print("---Result of data set split: ---")
	print("Train class distribution")
	total_train = Y_train.value_counts()[0] + Y_train.value_counts()[1]
	B_train = Y_train.value_counts()[0]
	M_train = Y_train.value_counts()[1]

	print(f"\tTotal: {total_train}")
	print(f"\tB(0): {B_train} ({B_train / total_train * 100:.1f}%)")
	print(f"\tM(1): {M_train} ({M_train / total_train * 100:.1f}%)")

	print("\nValidation class distribution")
	total_valid = Y_valid.value_counts()[0] + Y_valid.value_counts()[1]
	B_valid = Y_valid.value_counts()[0]
	M_valid = Y_valid.value_counts()[1]

	print(f"\tTotal: {total_valid}")
	print(f"\tB(0): {B_valid} ({B_valid / total_valid * 100:.1f}%)")
	print(f"\tM(1): {M_valid} ({M_valid / total_valid * 100:.1f}%)")

	print("\nTest class distribution")
	total_test = Y_test.value_counts()[0] + Y_test.value_counts()[1]
	B_test = Y_test.value_counts()[0]
	M_test = Y_test.value_counts()[1]

	print(f"\tTotal: {total_test}")
	print(f"\tB(0): {B_test} ({B_test / total_test * 100:.1f}%)")
	print(f"\tM(1): {M_test} ({M_test / total_test * 100:.1f}%)")
	return Y_train, X_train, Y_valid, X_valid, Y_test, X_test

def preprocess_data(data):
	cleaned = data.copy()
	
	#Drop the ID column, since it is not necessary for the training
	#Caution: After that column 1 stays column 1 and does not get column 0
	cleaned = cleaned.drop(columns=[0])

	#1. Impute missing in each column with the mean of that column
	#2. Standardize the values with mean and standard deviation
	for i in range(2, 31):
		#replace all 0 with a Nan value to calc the mean of each column without 0
		cleaned[i] = cleaned[i].replace(to_replace=0, value=np.nan)
		mean = cleaned[i].mean()
		std = cleaned[i].std()
		cleaned[i] = cleaned[i].replace(to_replace=np.nan, value=mean)

		#Standardizing the values.
		#With this step the imputed values are ofc 0 again, since we set tem to be the mean
		#and here we subtract the mean. Now they are centered.
		cleaned[i] = (cleaned[i] - mean)/std
	
	#Encode the categorial values M and B to 1 and 0
	cleaned[1] = cleaned[1].map({'M': 1, 'B': 0})
	
	return cleaned


#Import the data. The CSV-dataset has no Header line
df = pd.read_csv("./csv/data.csv", header=None)
print("-----------Unprocessed data: ------------")
print("Head: \n", df.head())
print("Tail: \n", df.tail())
print("Describe: \n", df.describe())
print("---Count of target in data set---")
print(df[1].value_counts())
print("---------------------------------\n")

#Preprocess the data:
# Impute missing values with mean
# Standardize the values
# Map M and B to 1 and 0
preprocessed_data = preprocess_data(df)
print("-----------Preprocessed data: ------------")
print("Head: \n", preprocessed_data.head())
print("Tail: \n", preprocessed_data.tail())
print("Describe: \n", preprocessed_data.describe())
print("---------------------------------\n")

os.makedirs('./csv/created/', exist_ok=True)
preprocessed_data.to_csv('./csv/created/data_cleaned.csv', index=False)

Y_train, X_train, Y_val, X_val, Y_test, X_test = train_val_test_split(preprocessed_data)

X_train.to_csv('./csv/created/features_train.csv', index=False)
X_val.to_csv('./csv/created/features_val.csv', index=False)
X_test.to_csv('./csv/created/features_test.csv', index=False)
Y_train.to_csv('./csv/created/target_train.csv', index=False)
Y_val.to_csv('./csv/created/target_val.csv', index=False)
Y_test.to_csv('./csv/created/target_test.csv', index=False)


#Plot a histogram
plt.hist(df.iloc[:, 1], bins=2)
plt.title("Distribution Target")
plt.show()