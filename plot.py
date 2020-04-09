import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


y = np.array([
    # Deaths
    6, 8, 21, 35, 55, 71, 103, 144, 177, 233, 281, 335, 422, 463,
    578, 759, 1019, 1228, 1408, 1789, 2352, 2921, 3605, 4313, 4934,
    5373, 6159, 7097
])

x = np.array([
    # Day
    1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30
])

labels = [
    # Labels corresponding to each day
    '10 Mar.', '12 Mar.', '14 Mar.', '15 Mar.', '16 Mar.', '17 Mar.',
    '18 Mar.', '19 Mar.', '20 Mar.', '21 Mar.', '22 Mar.', '23 Mar.',
    '24 Mar.', '25 Mar.', '26 Mar.', '27 Mar.', '28 Mar.', '29 Mar.',
    '30 Mar.', '31 Mar.', '1 Apr.', '2 Apr.', '3 Apr.', '4 Apr.',
    '5 Apr.', '6 Apr.', '7 Apr.', '8 Apr.'
]

x_diffs = x[1:]
diffs = []
for idx, val in enumerate(y):
    if idx+1 < len(y):
        diffs += [y[idx+1]-val]

diffs = np.array(diffs)

# ensure we have the correct number of points
assert(len(y) == len(x) == len(labels))


def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c


popt, pcov = curve_fit(exponential_func, x, y)
popt_diffs, pcov_diffs = curve_fit(exponential_func, x_diffs, diffs)


# work out R^2
def fit_function(x):
    return popt[0] * np.exp(popt[1] * x) + popt[2]


residuals = y - fit_function(x)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)


# work out R^2 for diffs
def fit_function_diffs(x):
    return popt_diffs[0] * np.exp(popt_diffs[1] * x) + popt_diffs[2]


residuals_diffs = diffs - fit_function_diffs(x_diffs)
ss_res_diffs = np.sum(residuals_diffs**2)
ss_tot_diffs = np.sum((diffs - np.mean(diffs))**2)
r_squared_diffs = 1 - (ss_res_diffs / ss_tot_diffs)

# Plot
plt.title(f'Covid-19 UK Hospitalised Fatalities', fontsize=16)
plt.plot(
    x, y,
    linestyle='-', label="Reported Deaths", marker='o',
    markersize='2', color="red", dash_joinstyle='bevel'
)

plt.plot(
    x, exponential_func(x, *popt),
    label=r'$y={a}e^{b}{c}$'.format(
        a=round(popt[0], 2),
        b='{' + str(round(popt[1], 2)) + 'x}',
        c=round(popt[2], 2),
    ),
    linestyle='--',
    color='grey',
    linewidth=1

)
plt.plot(
    [], [], ' ',
    label="$R^2={r_squared}$".format(r_squared=round(r_squared, 4))
)
plt.plot(
    x_diffs, diffs, '-o', markersize='2',
    color='orange', label='Deaths per day'
)
plt.plot(
    x, exponential_func(x, *popt_diffs),
    label=r'$y={a}e^{b}{c}$'.format(
        a=round(popt_diffs[0], 2),
        b='{' + str(round(popt_diffs[1], 2)) + 'x}',
        c=round(popt_diffs[2], 2),
    ),
    linestyle='--',
    color='blue',
    linewidth=1

)
plt.plot(
    [], [], ' ',
    label="$R^2={r_squared}$".format(r_squared=round(r_squared_diffs, 4))
)

# Annotate each point with its value
for a, b in zip(x, y):
    plt.text(a, b, str(b), fontsize=10, color='black')

for a, b in zip(x_diffs, diffs):
    plt.text(a, b, str(b), fontsize=10, color='black')

plt.grid(axis='y', which='major', color='#eeeeee', linestyle='-')
plt.xticks(x, labels, rotation='vertical')
plt.legend(loc='upper left')
plt.show()
