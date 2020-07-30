import matplotlib.pyplot as plt
import numpy as np
import json

# Sense-bert
fp = open("../data/gathered_accuracy_sense-bert.json")
data = json.load(fp)
fp.close()

x = []
y = []

x_bert = []
y_bert = []

color = []
color2 = []
for token in data:
    entry = data[token]
    if entry["name"].endswith("sparse"):
        color.append("green")
    else:
        color.append("blue")
    y.append(entry["name"])
    x.append(float(entry["scores"][0].split("@")[0])*100)
    print(f"{y[-1]}\t{x[-1]}")

# bert
fp = open("../data/gathered_accuracy_bert.json")
data = json.load(fp)
fp.close()

for token in data:
    entry = data[token]
    if entry["name"].endswith("sparse"):
        color2.append("red")
    else:
        color2.append("orange")
    y_bert.append(f"bert-large-uncased_layer_24_{entry['name']}")
    x_bert.append(float(entry["scores"][0].split("@")[0])*100)
    print(f"{y_bert[-1]}\t{x_bert[-1]}")

plt.rcdefaults()
fig, ax = plt.subplots()
fig.set_size_inches(12, 7)
# Example data
y_pos = np.arange(len(y))
y_pos2 = np.arange(len(y), len(y)+len(y_bert))

ax.barh(y_pos, x, align='center', color=color)
ax.barh(y_pos2, x_bert, align='center', color=color2)
ax.set_yticks(np.arange(len(y)+len(y_bert)))
ax.set_yticklabels(y+y_bert)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Accuracy Score (%)')
ax.set_title('Methods')

plt.show()