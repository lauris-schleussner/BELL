from matplotlib import pyplot as plt
import numpy
import pandas

'''
data = []
with open('E:/BELL/bellutils/singlelabeldist.txt') as f:
    for line in f:
        line = line.rstrip()
        data.append(line.split(" "))
'''
amount = []
style = []

data = []
with open('E:/BELL/bellutils/multilabeldist.txt') as f:
    for line in f:
        line = line.replace("(", "")
        line = line.replace(")", "")
        line = line.replace("'", "")
        line = line.replace(" ", "")
        line = line.rstrip()
        line = line.split(",")


        data.append([line[0] + " + " + line[1], int(line[2])])
        #print(line[3])

        #amount = int(line[3])


data = sorted(data, key=lambda l: int(l[1]), reverse=True)
c = 0
for i in data:
    c += 1
    if c >= 10:
        break

    style.append(i[0])
    amount.append(int(i[1]))

print(style)

plt.bar(style, amount)
plt.title("Häufigste Klassenkombinationen")
plt.xlabel("Kombinationen")
plt.ylabel("Anzahl")



plt.show()