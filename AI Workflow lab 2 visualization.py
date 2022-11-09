#!/usr/bin/env python
# coding: utf-8

# # IBM AI Workflow lab 2
# 
# This unit covers the topic of data visualization and this course is taught in Python. There are numerous frameworks out there and it is reasonable to use other languages, like R, to carry out data visualization. The reason for using matplotlib is that it is most common tool once you have accounted for direct and indirect usage. See the visualization below to get a better understanding how matplotlib fits into the Python landscape of visualization tools.

# In[1]:


import os
import re
import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt

plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

## specify the directory you saved the data and images in
DATA_DIR = os.path.join("D:\data_science\Data-Visualization-Unit-Local","data")
IMAGE_DIR = os.path.join("D:\data_science\Data-Visualization-Unit-Local","images")


# In[2]:


## load the data and print the shape
df = pd.read_csv(os.path.join(DATA_DIR, "world-happiness.csv"), index_col=0)
print("df: {} x {}".format(df.shape[0], df.shape[1]))

## clean up the column names
df.columns = [re.sub("\s+","_",col) for col in df.columns.tolist()]

## check the first few rows
df.head(n=4)


# In[3]:


## missing values summary
print("Missing Value Summary\n{}".format("-"*35))
print(df.isnull().sum(axis = 0))


# In[4]:


## drop the rows that have NaNs
print("Original Matrix:", df.shape)
df.dropna(inplace=True)
print("After NaNs removed:", df.shape)


# # The data
# 
# The original data are produced by the UN Sustainable Development Solutions Network (SDSN) and the report is compiled and available at https://worldhappiness.report. The following is the messaging on the report website:
# 
# The World Happiness Report is a landmark survey of the state of global happiness that ranks 156 countries by how happy their citizens perceive themselves to be. The report is produced by the United Nations Sustainable Development Solutions Network in partnership with the Ernesto Illy Foundation.
# 
# The World Happiness Report was written by a group of independent experts acting in their personal capacities. Any views expressed in this report do not necessarily reflect the views of any organization, agency or program of the United Nations.
# 
# so knowing this it makes sense to sort the data.

# In[5]:


df.sort_values(['Year', "Happiness_Score"], ascending=[True, False], inplace=True)
df.head(n=4)


# ## EDA and pandas

# In[6]:


columns_to_show = ["Happiness_Score","Health_(Life_Expectancy)"]
pd.pivot_table(df, index= 'Year', values=columns_to_show,aggfunc='mean').round(3)


# In[7]:


df.groupby(['Year'])[columns_to_show].mean().round(3)


# In[8]:


pd.pivot_table(df, index = ['Region', 'Year'], values=columns_to_show).round(3)


# Use pd.qcut() to bin the data by Happiness_Rank and create a pivot table that summarizes Happiness_Score and Health_(Life_Expectancy) with respect to Region.

# In[9]:


df['Happiness_Rank_bins'] = pd.qcut(df['Happiness_Score'], 5, labels = ['Very Unhappy', 'Unhappy', 'Happy/Unhappy', 'Happy', 'Very Happy'])

pd.pivot_table(df, index = ['Happiness_Rank_bins', 'Region'], values=columns_to_show).round(3).sort_values('Happiness_Score', ascending = False)


# ## Essentials of matplotlib

# In[10]:


fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

table1 = pd.pivot_table(df, index='Region', columns='Year', values="Happiness_Score")
table1.plot(kind='bar', ax=ax1)
ax1.set_ylabel("Happiness Score");

table2 = pd.pivot_table(df, index='Region', columns='Year', values="Health_(Life_Expectancy)")
table2.plot(kind='bar', ax=ax2)
ax2.set_ylabel("Health (Life_Expectancy)");

## adjust the axis to accomadate the legend
ax1.set_ylim((0,9.3))
ax2.set_ylim((0,1.3))


# In[11]:


## plot style, fonts and colors
plt.style.use('seaborn')

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16
COLORS = ["darkorange","royalblue","slategrey"]

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

DATA_DIR = os.path.join("D:/data_science/Data-Visualization-Unit-Local","data")
IMAGE_DIR = os.path.join("D:/data_science/Data-Visualization-Unit-Local","images")


def run_data_ingestion():
    """
    ready the data for EDA
    """

    print("... data ingestion")
    
    ## load the data and print the shape
    df = pd.read_csv(os.path.join(DATA_DIR,"world-happiness.csv"),index_col=0)

    ## clean up the column names
    df.columns = [re.sub("\s+","_",col) for col in df.columns.tolist()]

    ## drop the rows that have NaNs
    df.dropna(inplace=True)

    ## sort the data for more intuitive visualization
    df.sort_values(['Year', "Happiness_Score"], ascending=[True, False], inplace=True)

    return(df)

def create_subplot(table,ax):
    """
    create a subplot
    """

    table['average'] = (table[2015] + table[2016] + table[2017]) / 3
    regions = np.array(list(table.index))
    year_2015 = table[2015].values
    year_2016 = table[2016].values
    year_2017 = table[2017].values
    averages = table['average'].values
    sorted_inds = np.argsort(averages)

    ## make bar plot 
    N = regions.size
    ind = np.arange(N)
    width = 0.3
    rects1 = ax.bar(ind, year_2015[sorted_inds], width, color=COLORS[0], label='2015')
    rects2 = ax.bar(ind+width, year_2016[sorted_inds], width, color=COLORS[1], label='2016')
    rects3 = ax.bar(ind+width+width, year_2017[sorted_inds], width, color=COLORS[2], label='2017')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(regions[sorted_inds],rotation=90)
    ax.legend(loc='upper left')

def create_plot(df):
    """
    create a two panel subplot that summarizes two features of the data with respect to Year and Region
    """

    print("... creating plot")
    
    columns_to_show = ["Happiness_Score","Health_(Life_Expectancy)"]
    pd.pivot_table(df, index= 'Year',values=columns_to_show,aggfunc='mean').round(3)

    ## ready a figure
    fig = plt.figure(figsize=(9,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ## create first subplot
    table1 = pd.pivot_table(df,index='Region',columns='Year',values="Happiness_Score")
    create_subplot(table1,ax1)
    ax1.set_ylabel("Happiness Score")
    ax1.set_ylim((0,8.0))
    
    ## create second subplot
    table2 = pd.pivot_table(df,index='Region',columns='Year',values="Health_(Life_Expectancy)")
    create_subplot(table2,ax2)
    ax2.set_ylabel("Health (Life Expectancy)")
    ax2.set_ylim((0,1.0))
    
    ## ensure equal aspect ratio
    for ax in [ax1,ax2]:
        ax.set_aspect(1./ax.get_data_ratio()) 

    image_path = os.path.join(IMAGE_DIR,"happiness-summary.png")
    plt.savefig(image_path,bbox_inches='tight',pad_inches = 0,dpi=200)
    print("{} created.".format(image_path))
    
if __name__ == "__main__":

    df = run_data_ingestion()
    create_plot(df)


# In[12]:


get_ipython().system('python D:/data_science/Data-Visualization-Unit-Local/scripts/make-happiness-summary-plot.py')
Image(os.path.join(IMAGE_DIR, "happiness-summary.png"), width=800, height=600)


# ## Pair plots and correlation

# In[13]:


import seaborn as sns
sns.set(style="ticks", color_codes=True)

## make a pair plot
columns = ['Happiness_Score','Economy_(GDP_per_Capita)', 'Family', 'Health_(Life_Expectancy)',
           'Freedom', 'Trust_(Government_Corruption)']

axes = sns.pairplot(df,vars=columns,hue="Year",palette="husl")


# In[15]:


get_ipython().system('python D:/data_science/Data-Visualization-Unit-Local/scripts/make-happiness-corr-plot.py')
#IMAGE_DIR = os.path.join("D:\не говно\Data-Visualization-Unit-Local","images")
#IMAGE_DIR = os.path.join("D:\\ne-govno\\Data-Visualization-Unit-Local","images")
Image(os.path.join(IMAGE_DIR, "happiness-summary.png"), width=800, height=600)


# In[ ]:




