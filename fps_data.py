import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

CSV_FILENAME = "fps_ddpg.csv"
IMG_FILENAME = "fps_ddpg.png"
data = pd.read_csv(CSV_FILENAME)
mean_string = "Mean: " + str(round(data.mean()[1], 2))
std_string = "Std : " + str(round(data.std()[1], 2))

data.plot(x='seconds', y='fps')
plt.ylim(0, 40)
plt.title(CSV_FILENAME)
plt.xlabel("seconds")
plt.ylabel("steps per second")
plt.text(2, 4, mean_string)
plt.text(2, 2, std_string)
# plt.show()
plt.savefig(IMG_FILENAME)
