import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("./data.csv", header=None)

fig, ax = plt.subplots()
ax.plot(df[3], df[1], marker='o', linestyle='None')
plt.show()
