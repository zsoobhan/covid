#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

y = np.array([
    # Deaths
    6, 8, 21, 35, 55, 71, 103, 144, 177, 233,
    281, 335, 422, 463, 578, 759, 1019, 1228, 1408, 1789,
    2352, 2921, 3605, 4313, 4934, 5373, 6159, 7097, 7978, 8958,
    9875, 10612, 11329, 12107, 12868
])

x = np.array([
    # Day
    1, 3, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37
])

labels = [
    # Labels corresponding to each day
    '10 Mar.', '12 Mar.', '14 Mar.', '15 Mar.', '16 Mar.', '17 Mar.',
    '18 Mar.', '19 Mar.', '20 Mar.', '21 Mar.', '22 Mar.', '23 Mar.',
    '24 Mar.', '25 Mar.', '26 Mar.', '27 Mar.', '28 Mar.', '29 Mar.',
    '30 Mar.', '31 Mar.', '01 Apr.', '02 Apr.', '03 Apr.', '04 Apr.',
    '05 Apr.', '06 Apr.', '07 Apr.', '08 Apr.', '09 Apr.', '10 Apr.',
    '11 Apr', '12 Apr', '13 Apr.', '14 Apr.', '15 Apr.'
]

x_diffs = x[1:]
diffs = []
for idx, val in enumerate(y):
    if idx+1 < len(y):
        diffs += [y[idx+1]-val]

# ensure we have the correct number of points
assert(len(y) == len(x) == len(labels) == len(diffs)+1)


def exponential_func(x, a, b):
    return a * np.exp(b * x)


popt, pcov = curve_fit(exponential_func, x, y)
popt_diffs, pcov_diffs = curve_fit(exponential_func, x_diffs, diffs)


# work out R^2
def fit_function(x):
    return popt[0] * np.exp(popt[1] * x)


residuals = y - fit_function(x)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1.0 - (ss_res / ss_tot)


# Plot
plt.title(f'Covid-19 UK Hospitalised Fatalities', fontsize=16)
plt.plot(
    x, y,
    linestyle='-', label="Reported Deaths", marker='s',
    markersize='2', color="red", dash_joinstyle='bevel'
)

plt.plot(
    x, exponential_func(x, *popt),
    label=r'$y={a}e^{b}$'.format(
        a=round(popt[0], 2),
        b='{' + str(round(popt[1], 2)) + 'x}',
    ),
    linestyle='dashed',
    color='grey',
    linewidth=1

)
plt.plot(
    [], [], ' ',
    label="$R^2={r_squared}$".format(r_squared=round(r_squared, 4))
)
plt.plot(
    x_diffs, diffs, markersize='2',
    linestyle='-', marker='s',
    color='orange', label='Deaths per day'
)

# Annotate each point with its value
for a, b in zip(x, y):
    plt.text(a, b, str(b), fontsize=10, color='black')

for a, b in zip(x_diffs, diffs):
    plt.text(a, b, str(b), fontsize=9, color='black')

plt.grid(axis='y', which='major', color='#eeeeee', linestyle='-')
plt.xticks(x, labels, rotation='vertical')
plt.legend(loc='upper left')
plt.savefig('latest', dpi=200)
plt.show()
