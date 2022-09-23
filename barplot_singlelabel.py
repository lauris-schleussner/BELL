from matplotlib import pyplot as plt


data = []
with open('E:/BELL/bellutils/singlelabeldist.txt') as f:
    for line in f:
        line = line.rstrip()
        data.append(line.split(" "))

style = []
amount = []
data = sorted(data, key=lambda l: int(l[1]), reverse=True)
c = 0
for i in data:
    c += 1
    if c >= 6: # only take 5-1 classcombinations
        break

    labels = i[0]

    if labels == "Art_Nouveau_(Modern)":
        labels = "Art_Nouveau"

    style.append(labels)
    amount.append(int(i[1]))

print(style)

plt.bar(style, amount)
plt.title("Anzahl der Bilder pro Klasse")
plt.xticks(fontsize=13)

plt.show()