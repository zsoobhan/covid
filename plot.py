import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


y = np.array([
    6, 8, 21, 35, 55, 71, 103, 144, 177, 233, 281, 335, 422, 463,
    578, 759, 1019, 1228, 1408, 1789, 2352, 2921, 3605, 4313, 4934, 5373, 6159,
])

x = np.array([
    1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29
])

labels = [
    '10 Mar.', '12 Mar.', '14 Mar.', '15 Mar.', '16 Mar.', '17 Mar.',
    '18 Mar.', '19 Mar.', '20 Mar.', '21 Mar.', '22 Mar.', '23 Mar.',
    '24 Mar.', '25 Mar.', '26 Mar.', '27 Mar.', '28 Mar.', '29 Mar.',
    '30 Mar.', '31 Mar.', '1 Apr.', '2 Apr.', '3 Apr.', '4 Apr.',
    '5 Apr.', '6 Apr.', '7 Apr.',
]


def func(x, a, b):
    return a * np.exp(b * x)


popt, pcov = curve_fit(func, x, y)
perr = np.sqrt(np.diag(pcov))


# work out R^2
def fitfunc(x):
    return popt[0] * np.exp(popt[1] * x)


residuals = y - fitfunc(x)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)


plt.title(f'Covid-19 UK Deaths', fontsize=16)

plt.plot(x, y, 'ro', label="Deaths", marker='o', markersize='4')
plt.plot(
    x, func(x, *popt),
    label=r'${a}e^{b}$ and $R^2$={r_squared}'.format(
        a=round(popt[0], 2),
        b='{' + str(round(popt[1], 2)) + '}x',
        r_squared=round(r_squared, 4)
    )
)

for a, b in zip(x, y):
    plt.text(a, b, str(b), fontsize=10, color='black')

plt.grid(axis='y', which='major', color='#eeeeee', linestyle='-')
plt.xticks(x, labels, rotation='vertical')
plt.legend(loc='upper left')
plt.show()
