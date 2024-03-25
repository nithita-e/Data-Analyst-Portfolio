# import related package and dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

calendar = pd.read_csv('/calendar.csv')
listing = pd.read_csv('/listings.csv')
review = pd.read_csv('/reviews.csv')

# create new dataframe to split only rating columns
rating = listing[['review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']]

# calculate correlation for each rating score
sns.heatmap(rating.corr(),annot = True)
# --result turns to be positive, and the top three point that increase with overall rating is accuracy, cleanliness, and value 
# So, there is some clue that if the owner do the best on these three rating scores, they tend to get the higher overall rating, even they accommodation is not in the prime location,
# but they able to get higher rating as the accommodation in prime location.

# data prep
# cheack the outlier with boxplot
sns.boxplot(x = 'neighbourhood', y ='price', data = listing)
# -- there seems to have the outlier, so i use median, instead of mean.
price_street = listing.groupby('street', as_index = False)['price'].median().sort_values('price', ascending = True)

# set criteria and split data
median_price = price_street['price'].median()
bad_location = listing[listing['price'] < median_price]
great_location = listing[listing['price'] >= median_price]
bad_location_passed = bad_location[(bad_location['review_scores_accuracy'] >= 9) & (bad_location['review_scores_cleanliness'] >= 9) & (bad_location['review_scores_value'] >= 9)]

# print histogram, it seem to be similar, which is a good sign. -- but, correlation doesn't imply causation, so let's do hypothesis testing !
bad_location_passed.hist('review_scores_rating',color = 'blue')
great_location.hist('review_scores_rating',color = 'red')

# t-test hypothesis testing
# alpha = 0.05 
# HO : overall score of badlo_passed - overall score of great_lo = 0
# HA : overall score of badlo_passed - overall score of great-lo != 0 

from scipy.stats import t

x_bar_badlo_passed = bad_location_passed['review_scores_rating'].mean()
x_bar_greatlo = great_location['review_scores_rating'].mean()

std_badlo_passed = bad_location_passed['review_scores_rating'].std()
std_greatlo = great_location['review_scores_rating'].std()

n_badlo_passed = bad_location_passed['review_scores_rating'].count()
n_greatlo = bad_location_passed['review_scores_rating'].count()

nume = x_bar_badlo_passed - x_bar_greatlo
deno = np.sqrt(((std_badlo_passed**2)/n_badlo_passed) + ((std_greatlo**2)/n_greatlo))

t_stats = nume / deno

dof = (n_badlo_passed + n_greatlo) - 2

p_val = 2 * (1 - t.cdf(t_stats,df = dof))
print(p_val)

# failed to reject H0, conclude that there is no significance difference between two groups.
# Sum-up : there is a correlation the value, cleanliness, and accuracy have the positive correlation to the overall rate score (this sum up from the overall correlation of each score), 
# but there is no any causation that if you did best on value, cleanliness, and accuracy you will get more overall rating score.
