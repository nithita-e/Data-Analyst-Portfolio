# Start your code here!
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


soccer_wm = pd.read_csv('women_results.csv',parse_dates=['date'])
soccer_m = pd.read_csv('men_results.csv',parse_dates=['date'])

# filter
soccer_wm_filtered = soccer_wm[(soccer_wm['tournament'] == 'FIFA World Cup') & (soccer_wm['date'] > '2002-01-01')]
soccer_m_filtered = soccer_m[(soccer_m['tournament'] == 'FIFA World Cup') & (soccer_m['date'] > '2002-01-01')]

# mutate score
soccer_wm_filtered['total_score'] = soccer_wm_filtered['home_score'] + soccer_wm_filtered['away_score']
soccer_m_filtered['total_score'] = soccer_m_filtered['home_score'] + soccer_m_filtered['away_score']


# preview
soccer_wm_filtered.head()

#plt.hist(data = soccer_wm_filtered, x = 'total_score')
#plt.hist(data = soccer_m_filtered, x = 'total_score')

# try bootstrap 
sample_wm = soccer_wm_filtered.sample(frac = 0.5, replace = True)
sample_wm_bst = []
for i in range(1000):
    sample_wm_bst.append(
        np.mean(sample_wm.sample(frac = 1, replace = True)['total_score'])
    )
    
#plt.hist(sample_wm_bst)

# independence t-test
x_bar_wm = soccer_wm_filtered['total_score'].mean()
x_bar_m = soccer_m_filtered['total_score'].mean()

std_wm = soccer_wm_filtered['total_score'].std()
std_m = soccer_m_filtered['total_score'].std()

n_wm = soccer_wm_filtered['total_score'].count()
n_m = soccer_m_filtered['total_score'].count()

numerator = x_bar_wm - x_bar_m
denominator = np.sqrt(((std_wm ** 2) / n_wm) + ((std_m ** 2) / n_m))

t_stat = numerator / denominator

dof = (n_wm + n_m) -2

p_val = 2 * (1 - t.cdf(t_stat, df = dof))
print(p_val)
result = 'reject'

result_dict = {"p_val" : p_val, "result" : result }
