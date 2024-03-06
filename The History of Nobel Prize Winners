# Loading in required libraries
import pandas as pd
import seaborn as sns
import numpy as np

### Loading csv
nobel = pd.read_csv('data/nobel.csv')

### What is the most commonly awarded gender and birth country? Storing the string answers as top_gender and top_country ?
top_gender = nobel['sex'].value_counts().index[0]
top_country = nobel['birth_country'].value_counts().index[0]

### What decade had the highest proportion of US-born winners? Store this as an integer called max_decade_usa
# add is_usa column in existing dataframe
nobel['is_usa'] = nobel['birth_country'] == 'United States of America'
# turn year to be a decade column
nobel['decade'] = ((np.floor(nobel['year']/10)) * 10).astype(int)
max_decade_usa = nobel.groupby('decade', as_index = False)['is_usa'].mean().sort_values('is_usa',ascending = False).values[0]

### What decade and category pair had the highest proportion of female laureates? 
nobel['is_female'] = nobel['sex'] == 'Female'
prob = nobel.groupby(['decade','category'],as_index = False)['is_female'].mean().sort_values('is_female', ascending = False)
max_female_dict = {prob['decade'].values[0]:prob['category'].values[0]}

### Who was the first woman to receive a Nobel Prize, and in what category? 
first_female = nobel[nobel['is_female'] == True].sort_values('year')
first_woman_name = first_female['full_name'].values[0]
first_woman_category = first_female['category'].values[0]

### Which individuals or organizations have won multiple Nobel Prizes throughout the years?
nobel_count = nobel['full_name'].value_counts()
repeat_list = list(nobel_count[nobel_count >= 2].index)
