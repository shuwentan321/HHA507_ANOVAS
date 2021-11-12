# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:59:05 2021

@author: ikima
"""

#Import needed packages
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import levene, shapiro


#Import dataset to be analyzed
#Heart Failure Prediction, this dataset contains 12 clinical features for predicting death events.
mental_music = pd.read_csv('C:/Users/ikima/Downloads/archive/master_song_data.csv')

#Generate a list of columns to identify variables for the 1-way ANOVA tests
list(mental_music)

#Variables to be used for 1-way ANOVA tests
#Dependent variable (continous value) = 'Total_mental_health'
#Independent variable 1 (categorical value) = 'Audio_class'
#Independent variable 2 (categorical value) = 'Audio + Lyrics analysis'
#Independent variable 3 (categorical value) = 'Artist'

#Rename column names to avoid white space errors and better represent data
mental_music = mental_music.rename(columns={'Total_mental_health':'mental_health_score'})
mental_music = mental_music.rename(columns={'Audio + Lyrics analysis':'Audio_Lyrics_Analysis'})


#1 factor of Audio_class we have 4 levels
mental_music.Audio_class.value_counts()
len(mental_music.Audio_class.value_counts())

#1 factor of Audio_Lyrics_Analysis we have 4 levels
mental_music.Audio_Lyrics_Analysis.value_counts()
len(mental_music.Audio_Lyrics_Analysis.value_counts())

#1 factor of Artist we have 125 levels
mental_music.Artist.value_counts()
len(mental_music.Artist.value_counts())

#Create new dataframe with variables used for 1-way ANNOVA tests
df = mental_music[['mental_health_score','Audio_class','Audio_Lyrics_Analysis','Artist']]

#Conduct assumption testing
#Normality Tests using ANOVA framework
model = smf.ols('mental_health_score ~ C(Audio_class)', data= df).fit()
stats.shapiro(model.resid)
#Output is Shapiro Result(statistic=0.9451541304588318, p-value=4.7223892352121766e-07)

model2 = smf.ols('mental_health_score ~ C(Audio_Lyrics_Analysis)', data= df).fit()
stats.shapiro(model2.resid)
#Shapiro-Result(statistic=0.9448325037956238, p-value=4.408819904710981e-07)

model3 = smf.ols('mental_health_score ~ C(Artist)', data= df).fit()
stats.shapiro(model3.resid)
#Shapiro-Result(statistic=0.8074126243591309, p-value=3.167144916485054e-15)
#The null hypothesis for Shapiro-Wilk test is that the data has a normal distribution.
#If P-value is less than 0.05, we reject the null hypothesis and conclude the data is non-normal.
#P-value of less than 0.05 indicates a high likelihood that assumption for normality is not met for all three independent variables. 

#Graphical tests for normality
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)
normality_plot1, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of regression residuals \n with R value")
ax.set
#The result of R^2 = 0.9452 indicates a strong positive linear relationship.
#By looking at the plot, we can see that there are outliers within the data.
#The deviations at either end of the plot indicates that the data is skewed.

normality_plot2, stat = stats.probplot(model2.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of regression residuals \n with R value")
ax.set
#The result of R^2 = 0.9448 indicates a strong positive linear relationship.
#By looking at the graphs, we can see that there are outliers within the data.
#The deviations at either end of the plot indicates that the data is skewed.

normality_plot3, stat = stats.probplot(model3.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of regression residuals \n with R value")
ax.set
#The result of R^2 = 0.7994 indicates a weak positive linear relationship.
#By looking at the graphs, we can see that there are outliers within the data.

#Test for homogeneity of variances
#Since we indicated with the shapiro test that the data is not normally distributed,
#we will be using the levene test to test for homogeneity of variances.

#Use label encoding to encode data to avoid conversion error from string to float.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)

#Recast the series to float
df['mental_health_score'] = df['mental_health_score'].astype('float') 
df['Audio_class'] = df['Audio_class'].astype('float')

#Perform levene test
stats.levene(df['mental_health_score'], df['Audio_class'])
#Levene- Result(statistic=188.05412102155282, p-value=1.710543544952837e-35)
#P-value < 0.05, indicating unequal variance.

stats.levene(df['mental_health_score'], df['Audio_Lyrics_Analysis']) 
#Levene Result(statistic=144.34642086878696, p-value=1.0716706030648318e-28)
#P-value < 0.05, indicating unequal variance.

stats.levene(df['mental_health_score'], df['Artist']) 
#Levene Result(statistic=238.93556429694604, p-value=8.50786973920798e-43)
#P-value < 0.05, indicating unequal variance.

#Perform one-way ANOVA tests on non-parametric data

#Q: Is there difference between 'levels' of audio class and mental health score?
stats.kruskal(df['mental_health_score'], df['Audio_class'])
#Kruskal Result(statistic=155.04194102636257, p-value=1.3709713791337518e-35)
#P-value < 0.05 indicates that there is significant difference between 'levels' of audio class and mental health score.

#Q: Is there difference between 'levels' of audio lyrics analysis and mental health score?
stats.kruskal(df['mental_health_score'], df['Audio_Lyrics_Analysis'])
#Kruskal Result(statistic=124.81651406142402, p-value=5.582516355332212e-29)
#P-value < 0.05 indicates that there is significant difference between 'levels' of audio lyrics analysis and mental health score.

#Q: Is there difference between 'levels' of artist and mental health score?
stats.kruskal(df['mental_health_score'], df['Artist'])
#Kruskal Result(statistic=256.3302244238847, p-value=1.0825827034742911e-57)
#P-value < 0.05 indicates that there is significant difference between 'levels' of artist and mental health score.

#Perform post-hoc tests
import pingouin as pg

pg.pairwise_gameshowell(data=df, dv='mental_health_score', between='Audio_class').round(3)
#Output
"""     A    B  mean(A)  mean(B)   diff     se      T       df   pval  hedges
0  0.0  1.0    6.897    6.375  0.522  0.894  0.583   53.612  0.900   0.136
1  0.0  2.0    6.897    5.321  1.576  0.829  1.902   44.007  0.242   0.396
2  0.0  3.0    6.897    6.913 -0.016  1.221 -0.014   43.256  0.900  -0.004
3  1.0  2.0    6.375    5.321  1.054  0.633  1.665  100.381  0.348   0.288
4  1.0  3.0    6.375    6.913 -0.538  1.098 -0.490   34.348  0.900  -0.123
5  2.0  3.0    5.321    6.913 -1.592  1.045 -1.524   29.050  0.438  -0.348"""

pg.pairwise_gameshowell(data=df, dv='mental_health_score', between='Audio_Lyrics_Analysis').round(3)
#Output
"""   A  B  mean(A)  mean(B)   diff     se      T      df   pval  hedges
0  0  1    6.875    6.929 -0.054  1.205 -0.044  46.370  0.900  -0.012
1  0  2    6.875    6.447  0.428  1.065  0.402  37.267  0.900   0.100
2  0  3    6.875    5.299  1.576  1.006  1.566  30.924  0.413   0.352
3  1  2    6.929    6.447  0.482  0.919  0.524  50.794  0.900   0.124
4  1  3    6.929    5.299  1.630  0.851  1.916  41.143  0.238   0.404
5  2  3    6.447    5.299  1.148  0.636  1.805  96.852  0.277   0.314"""


pg.pairwise_gameshowell(data=df, dv='mental_health_score', between='Artist').round(3)