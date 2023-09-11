import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import os
import glob

ROOT_DIR = pathlib.Path(__file__).parent
ANALYZE_DIR = os.path.join(ROOT_DIR, "data/outputs/analyze_data/can")

train_loss_file = glob.glob(os.path.join(ANALYZE_DIR, "*rate.csv"))
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
colors = ['red', 'blue', 'yellow']
labels = ["ACT", "DP", "IBC"]
for idx, path in enumerate(train_loss_file):
    df = pd.read_csv(path)
    col = df.columns
    print(col[0], col[4])
    if "ACT" in path:
        df.plot(kind='line', x=col[0], y=col[4], title='ACT vs. DP vs. IBC', ax=ax, label="ACT", color="blue")
        ax.fill_between(df[col[0]], df[col[4]], color="blue", alpha=0.3)
    elif "IBC" in path:
        df.plot(kind='line', x=col[0], y=col[4], title='ACT vs. DP vs. IBC', ax=ax, label="IBC", color="yellow")
        ax.fill_between(df[col[0]], df[col[4]], color="yellow", alpha=0.3)
    else:
        df.plot(kind='line', x=col[0], y=col[4], title='ACT vs. DP vs. IBC', ax=ax, label="DP", color="red")
        ax.fill_between(df[col[0]], df[col[4]], color="red", alpha=0.3)
    print(f"y length: {len(df[col[4]])}")

ax.set_xlabel('EPOCH')
custom_xticks = [0, 20, 40, 60, 80, 100]
ax.set_xticks(custom_xticks)
ax.set_xlim(0, df[col[0]].values[-1])

ax.set_ylabel('SUCCESS RATE')
# custom_yticks = [0, 0.1, 0.2, 0.3, 0.4]
# ax.set_yticks(custom_yticks)
ax.set_ylim(0,1.1)

ax.legend()
plt.tight_layout()
plt.show()