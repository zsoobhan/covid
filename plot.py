#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

y = np.array([
    # Deaths
    6, 8, 21, 35, 55, 71, 103, 144, 177, 233,
    281, 335, 422, 463, 578, 759, 1019, 1228, 1408, 1789,
    2352, 2921, 3605, 4313, 4934, 5373, 6159, 7097, 7978, 8958,
    9875, 10612, 11329, 12107, 12868, 13729, 14576, 15464, 16060,
    16509, 17337, 18100, 18738, 19506, 20319, 20732, 21092, 21678,
    26097, 26771, 27510, 28131, 28446, 28734, 29427, 30076, 30615,
    31241, 31587, 31855, 32065, 32692, 33186, 33614, 33998, 34466,
    34636, 34796, 35341
])

x = np.array([
    # Day
    1, 3, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71


])

labels = [
    # Labels corresponding to each day
    '10 Mar', '12 Mar', '14 Mar', '15 Mar', '16 Mar', '17 Mar',
    '18 Mar', '19 Mar', '20 Mar', '21 Mar', '22 Mar', '23 Mar',
    '24 Mar', '25 Mar', '26 Mar', '27 Mar', '28 Mar', '29 Mar',
    '30 Mar', '31 Mar', '01 Apr', '02 Apr', '03 Apr', '04 Apr',
    '05 Apr', '06 Apr', '07 Apr', '08 Apr', '09 Apr', '10 Apr',
    '11 Apr', '12 Apr', '13 Apr', '14 Apr', '15 Apr', '16 Apr',
    '17 Apr', '18 Apr', '19 Apr', '20 Apr', '21 Apr', '22 Apr',
    '23 Apr', '24 Apr', '25 Apr', '26 Apr', '27 Apr', '28 Apr',
    '29 Apr', '30 Apr', '01 May', '02 May', '03 May', '04 May',
    '05 May', '06 May', '07 May', '08 May', '09 May', '10 May',
    '11 May', '12 May', '13 May', '14 May', '15 May', '16 May',
    '17 May', '18 May', '19 May'

]

x_diffs = x[1:]
diffs = []
for idx, val in enumerate(y):
    if idx+1 < len(y):
        diffs += [y[idx+1]-val]

# ensure we have the correct number of points
assert(len(y) == len(x) == len(labels) == len(diffs)+1)


# Plot
plt.title(f'Covid-19 UK Fatalities', fontsize=16)
plt.plot(
    x, y,
    linestyle='-', label="Reported Deaths", marker='s',
    markersize='2', color="red", dash_joinstyle='bevel'
)

plt.plot(
    x_diffs, diffs, markersize='2',
    linestyle='-', marker='s',
    color='orange', label='Deaths per day'
)

# Annotate each point with its value
for a, b in zip(x, y):
    plt.text(a, b, str(b), fontsize=7, color='black')

for a, b in zip(x_diffs, diffs):
    plt.text(a, b, str(b), fontsize=6, color='black')

ax = plt.gca()

# annotations
ax.axvline(x=50, alpha=0.7, color='blue', linestyle='--', linewidth=1)
plt.text(
    50, 10000, '$\\rightarrow$ \n *includes \n non-hospital \n deaths',
    alpha=0.7, color='blue', fontsize=5
)

ax.axvline(x=63, alpha=0.7, color='blue', linestyle='--', linewidth=1)
plt.text(
    63, 10000, '$\\rightarrow$\n *stay home\n\t$\\downarrow$\n stay alert',
    alpha=0.7, color='blue', fontsize=5,
)


plt.grid(axis='y', which='major', color='#eeeeee', linestyle='-')
plt.xticks(x[::2], labels[::2], rotation='vertical', fontsize='8')
plt.legend(loc='upper left')
plt.savefig('latest', dpi=200)
plt.show()
