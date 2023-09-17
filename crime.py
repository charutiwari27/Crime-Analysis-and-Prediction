#ML Mini Project
#ASSIGNMENT 8

print("ASSIGNMENT 8 : MINI PROJECT")
print("CRIME ANALYSIS AND PREDICTION")
print("1609 : Monali Borekar\n1611 : Felicia Carvalho\n1613 : Charu
Tiwari\n1614 : Vedika Chavan")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
!pip install prophet
import prophet
from prophet import Prophet

#reading csv file
crimes_df = pd.read_csv('crime.csv', error_bad_lines=False)
crimes_df = crimes_df[['DAY','MONTH','YEAR','TYPE','NEIGHBOURHOOD']]
print("\nDataset:")
print(crimes_df.head())

#CONVERTING SEPERATE DAY,MONTH,YEAR INTO DATE
dff = crimes_df.loc[:,['YEAR','MONTH','DAY']]
dff['YEAR'] = dff['YEAR'].map(str)
dff['MONTH'] = dff['MONTH'].map(str)
dff['DAY'] = dff['DAY'].map(str)
dff["DATE"] = pd.to_datetime(dff['YEAR'] + "-" + dff['MONTH'] + "-" +
dff['DAY'])
dff = dff["DATE"]
crimes_df['DATE'] = dff#adding DATE column to csv
crimes_df.DATE = pd.to_datetime(crimes_df.DATE, format='%m/%d/%Y %I:%M:%S
%p')#formatting the date

crimess_df = crimes_df.copy()
crimess_df.index = pd.DatetimeIndex(crimess_df.DATE)#date as index
crimes_df =
crimes_df[['DATE','DAY','MONTH','YEAR','TYPE','NEIGHBOURHOOD']]#datarframe
with specific columns
crimess_df =
crimess_df[['DATE','DAY','MONTH','YEAR','TYPE','NEIGHBOURHOOD']]
#printing first 20 values
print("\nDataset after additing DATE column:")
print(crimes_df.head(20))

print("\n--------------------------------------------")
print("1. Data Visualization")
print("2. Analizing a particular neighbourhood")
print("3. Chi-Square test")
print("4. Forecasting using Prophet")
print("--------------------------------------------")

#1
print("\n\n1.**********DATA VISUALIZATION**********")
#OVERALL
y1 = crimes_df.copy()
y1 = y1[['YEAR']]#year df
m1 = crimes_df.copy()
m1 = m1[['MONTH']]#month df
#grouping and counting
#A groupby operation involves some combination of splitting the object,
applying a function, and combining the results. This can be used to group
large amounts of data and compute operations on these groups.
y1['COUNT'] = y1.groupby('YEAR')['YEAR'].transform('count')
m1['COUNT'] = m1.groupby('MONTH')['MONTH'].transform('count')
#crime per year
print("\nYEAR-WISE CRIME")


print(y1.drop_duplicates())
plt.plot(crimess_df.resample('Y').size())
plt.title('Crimes Count Per Year')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')
plt.show()
#bargraph
plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'YEAR', data = crimes_df)
for p in ax.patches:
ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25,
p.get_height()+0.01))
plt.show()
#crime per month
print("\nMONTH-WISE CRIME")
print(m1.drop_duplicates())
plt.plot(crimess_df.resample('M').size())
plt.title('Crimes Count Per Month')
plt.xlabel('Months')
plt.ylabel('Number of Crimes')
plt.show()
#bargraph
plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'MONTH', data = crimes_df)
for p in ax.patches:
ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25,
p.get_height()+0.01))
plt.show()

#type
print("\n")
print("\nTYPE-WISE CRIME")
plt.figure(figsize = (15, 10))
ax = sns.countplot(y = 'TYPE', data = crimes_df)
for p in ax.patches:
ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25,
p.get_height()+0.01))
plt.show()

#neighbourhood
print("\n")
print("\nNEIGHBOURHOOD-WISE CRIME")
plt.figure(figsize = (15, 10))
ax = sns.countplot(y = 'NEIGHBOURHOOD', data = crimes_df)
for p in ax.patches:
ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25,
p.get_height()+0.01))
plt.show()

#2
print("\n\n2.**********NEIGHBOURHOOD ANALYSIS**********")
neigh = crimes_df[['NEIGHBOURHOOD']]
print("List of Neighbourhoods:")
neigh = neigh.drop_duplicates()
neigh = neigh.dropna()
print(neigh)#df of neighbourhoods
c = input("\nEnter a neighbourhood:")#accepting neighbourhood
c1=crimes_df[crimes_df['NEIGHBOURHOOD']==c]
c2=crimes_df[crimes_df['NEIGHBOURHOOD']==c]
print("\nCrimes in",c)
c1['COUNT'] = c1.groupby('TYPE')['TYPE'].transform('count')
c2['COUNT'] = c1.groupby('MONTH')['MONTH'].transform('count')
c2 = c2[['MONTH','COUNT']]
c2 = c2.drop_duplicates()
c1 = c1[['TYPE','COUNT']]
c1 = c1.drop_duplicates()
print(c1)
print("The most common crime is :",c1.iloc[0,0])#common crime
#plotting bar
c_name = np.array(c1['TYPE'])
c_count = np.array(c1['COUNT'])
#plt.bar(c_name,c_count)

#piechart
plt.pie(c_count,labels = c_name)
plt.title("Crimes vs Count in the neighbourhood ")
plt.show()

#3
print("\n\n3.**********CHI - SQUARE TEST**********")
print("\nMonth-wise crime in",c)
print(c2)
print("\nCHI-SQUARE TEST")
!pip install scipy
from scipy.stats import chi2_contingency
a1 = np.array(c2['MONTH'])
a2 = np.array(c2['COUNT'])
# defining the table
data = [[a1,a2]]
stat, p, dof, expected = chi2_contingency(data)
# interpret p-value
alpha = 0.05
print("Threshold:",alpha)
print("The value of is " + str(p))
if p <= alpha:
print('Month and No. of crimes are related to each other')
else:
print('Independent (H0 holds true)')
plt.bar(a1,a2)
plt.title("Month vs CrimeCount")
plt.xlabel("Month")
plt.ylabel("Count")
plt.show()

#4
print("\n\n4.**********FORECASTING**********")

c_prophet = crimess_df.resample('M').size().reset_index()
c_prophet.columns = ['Date', 'Crime Count']
c_prophet_df = pd.DataFrame(c_prophet)
print(c_prophet)
c_prophet_df_final = c_prophet_df.rename(columns={'Date':'ds', 'Crime
Count':'y'})
print(c_prophet_df_final)
m = Prophet()#object
m.fit(c_prophet_df_final)
# Forecasting into the future
future = m.make_future_dataframe(periods=1825)
forecast = m.predict(future)
figure = m.plot(forecast, xlabel='Date', ylabel='Crime Rate')
figure3 = m.plot_components(forecast)

from sklearn import preprocessing
crimes_df = pd.read_csv('crime.csv', error_bad_lines=False)
le = preprocessing.LabelEncoder()
crimes2 = crimes_df.copy()
crimes2['TYPE_ENCODED'] = le.fit_transform(crimes2['TYPE'])
crimes2['NEIGHBOURHOOD_ENCODED'] =
le.fit_transform(crimes2['NEIGHBOURHOOD'])
df_t = crimes2[['TYPE_ENCODED','TYPE']]
df_t = df_t.drop_duplicates()
print(df_t)
df_n = crimes2[['NEIGHBOURHOOD_ENCODED','NEIGHBOURHOOD']]
df_n = df_n.drop_duplicates()
print(df_n)
prediction_df =
crimes2[['DAY','MONTH','YEAR','NEIGHBOURHOOD_ENCODED','TYPE_ENCODED']]
X = prediction_df[['DAY','MONTH','YEAR','NEIGHBOURHOOD_ENCODED']].values

y = prediction_df['TYPE_ENCODED'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =
0.05)

#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size =
0.00012)
model = tree.DecisionTreeClassifier()
model.fit(X_train2, y_train2)

print("\nPredicting crime based on date and neighbourhood")
d = int(input("Enter day:"))
m = int(input("Enter month:"))
y = int(input("Enter year:"))
n = int(input("Enter neighbourhood:"))
arr = np.array([[d,m,y,n]])
y_pred22 = model.predict(arr)
df2 = df_t[df_t['TYPE_ENCODED']==y_pred22[0]]
print("The predicted crime is: ",df2.iloc[0,1])