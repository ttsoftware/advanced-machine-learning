import numpy as np

f = open('../../data/subject1_csv/eeg_200605191428_epochs/tiny.csv', 'r')
lines = f.readlines()
f.close()
rows = np.array([[float(x) for x in line.split(',')] for line in lines])
columns = rows.T

means = np.mean(columns, axis=tuple(range(1, 2)))

x_ratio = [0.95, 1, 0.95, 0.9, 0.8, 0.6, 0.4, 0.225, 0.2, 0.175, 0.15, 0.1, 0.05, 0.025, 0.02, 0.015, 0.01, 0.005, 0.001, 0]
y_ratio = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 1, 1, 0.9, 0.8, 0.6, 0.2, 0.125]

multi_ratio = np.array([[x * y for y in y_ratio] for x in x_ratio])

artifacts = []

for x in range(len(x_ratio)):
    artifacts.append([])
    for y in range(len(y_ratio)):
        artifacts[x].append(rows[x+20][y] + means[y] * multi_ratio[x][y])

artifacts = [','.join(map(lambda number: str(number), row))+'\n' for row in artifacts]

lines = lines[:20] + artifacts

f = open('../../data/subject1_csv/eeg_200605191428_epochs/tiny_artifacts.csv', 'w')

for x in range(len(lines)):
    f.write(lines[x])
