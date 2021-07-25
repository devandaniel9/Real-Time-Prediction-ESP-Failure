#Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.stats import linregress
import streamlit as st
import plotly.express as px
from sklearn.metrics import accuracy_score

#Title and Input
st.title('Real Time Prediction of ESP Failure')

new_list = []

legend = [  'Amps',
            'Intake Pressure',
            'Discharge Pressure',
            'Intake Temperature',
            'Motor Temperature',
            'Vibration',
            'Rate']

st.sidebar.header('Current Data Input')
for i in legend:
    new_list.append(st.sidebar.number_input('Input {} '.format(i), min_value=0.00))


#Read the Dataset
data = pd.read_excel("dataset.xlsx")

#Split into input (X) and output (y)
x = data.drop("TRIP",axis = 1)
y = data.TRIP

#Implement SMOTE algorithm
sm = SMOTE(sampling_strategy='not majority', k_neighbors=1,random_state=123)
x_res,y_res = sm.fit_resample(x,y)

#Split data to train and test data
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.25, random_state=4)

#Initialize, train, and test the model
logistic_regression = LogisticRegression(random_state=123)
logistic_regression.fit(x_train,y_train)
y_pred = logistic_regression.predict(x_test)

#Read ESP Input Data
input = pd.read_excel("input.xlsx")

##Three Days Processing
n3d = input
#Head of data
head3d = n3d.head(1)
head3d_A = head3d.A
head3d_IP = head3d.IP
head3d_DP = head3d.DP
head3d_IT = head3d.IT
head3d_MT = head3d.MT
head3d_V = head3d.V
head3d_R = head3d.R

#Set x and y data for Three Days Processing
x3d = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8, 9/8, 10/8, 11/8, 12/8, 13/8, 14/8, 
        15/8, 16/8, 17/8, 18/8, 19/8, 20/8, 21/8, 22/8, 23/8, 24/8]
x3d_TIME = n3d.TIME.astype(str).values.tolist()
y3d_A = n3d.A
y3d_IP = n3d.IP
y3d_DP = n3d.DP
y3d_IT = n3d.IT
y3d_MT = n3d.MT
y3d_V = n3d.V
y3d_R = n3d.R
#Average Ampere
slope3d_A, intercept3d_A, rvalue3d_A, pvalue3d_A, stderr3d_A = linregress(x3d, y3d_A)
processed3d_A = float((slope3d_A) / (head3d_A))
#Intake Pressure
slope3d_IP, intercept, rvalue, pvalue, stderr = linregress(x3d, y3d_IP)
processed3d_IP = float((slope3d_IP) / (head3d_IP))
#Discharge Pressure
slope3d_DP, intercept, rvalue, pvalue, stderr = linregress(x3d, y3d_DP)
processed3d_DP = float((slope3d_DP) / (head3d_DP))
#Intake Temperature
slope3d_IT, intercept, rvalue, pvalue, stderr = linregress(x3d, y3d_IT)
processed3d_IT = float((slope3d_IT) / (head3d_IT))
#Motor Temperature
slope3d_MT, intercept, rvalue, pvalue, stderr = linregress(x3d, y3d_MT)
processed3d_MT = float((slope3d_MT) / (head3d_MT))
#Vibration
slope3d_V, intercept, rvalue, pvalue, stderr = linregress(x3d, y3d_V)
processed3d_V = float((slope3d_V) / (head3d_V))
#Rate
slope3d_R, intercept, rvalue, pvalue, stderr = linregress(x3d, y3d_R)
processed3d_R = float((slope3d_R) / (head3d_R))
#3 days dataframe
data3d = { 'A':[processed3d_A], 
          'IP':[processed3d_IP], 
          'DP':[processed3d_DP], 
          'IT':[processed3d_IT], 
          'MT':[processed3d_MT], 
          'V':[processed3d_V], 
          'R':[processed3d_R] }
df3d = pd.DataFrame(data3d)
#3 days prediction
new_prediction3d = logistic_regression.predict(df3d)
##One Day Processing
n1d = input.tail(9)
#Head of data
head1d = n1d.head(1)
head1d_A = head1d.A
head1d_IP = head1d.IP
head1d_DP = head1d.DP
head1d_IT = head1d.IT
head1d_MT = head1d.MT
head1d_V = head1d.V
head1d_R = head1d.R

#Set x and y data for One Day Processing
x1d = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8, 9/8]
y1d_A = n1d.A
y1d_IP = n1d.IP
y1d_DP = n1d.DP
y1d_IT = n1d.IT
y1d_MT = n1d.MT
y1d_V = n1d.V
y1d_R = n1d.R
#Average Ampere
slope1d_A, intercept1d_A, rvalue1d_A, pvalue1d_A, stderr1d_A = linregress(x1d, y1d_A)
processed1d_A = float((slope1d_A) / (head1d_A))
#Intake Pressure
slope1d_IP, intercept, rvalue, pvalue, stderr = linregress(x1d, y1d_IP)
processed1d_IP = float((slope1d_IP) / (head1d_IP))
#Discharge Pressure
slope1d_DP, intercept, rvalue, pvalue, stderr = linregress(x1d, y1d_DP)
processed1d_DP = float((slope1d_DP) / (head1d_DP))
#Intake Temperature
slope1d_IT, intercept, rvalue, pvalue, stderr = linregress(x1d, y1d_IT)
processed1d_IT = float((slope1d_IT) / (head1d_IT))
#Motor Temperature
slope1d_MT, intercept, rvalue, pvalue, stderr = linregress(x1d, y1d_MT)
processed1d_MT = float((slope1d_MT) / (head1d_MT))
#Vibration
slope1d_V, intercept, rvalue, pvalue, stderr = linregress(x1d, y1d_V)
processed1d_V = float((slope1d_V) / (head1d_V))
#Rate
slope1d_R, intercept, rvalue, pvalue, stderr = linregress(x1d, y1d_R)
processed1d_R = float((slope1d_R) / (head1d_R))
#1 day dataframe
data1d = { 'A':[processed1d_A], 
        'IP':[processed1d_IP], 
        'DP':[processed1d_DP], 
        'IT':[processed1d_IT], 
        'MT':[processed1d_MT], 
        'V':[processed1d_V], 
        'R':[processed1d_R] }
df1d = pd.DataFrame(data1d)
#1 day prediction
new_prediction1d = logistic_regression.predict(df1d)

##Three Hours Processing
n3h = input.tail(2)
#Head of data
head3h = n3h.head(1)
head3h_A = head3h.A
head3h_IP = head3h.IP
head3h_DP = head3h.DP
head3h_IT = head3h.IT
head3h_MT = head3h.MT
head3h_V = head3h.V
head3h_R = head3h.R
#Set x and y data for One Day Processing
x3h = [1/8, 2/8]
y3h_A = n3h.A
y3h_IP = n3h.IP
y3h_DP = n3h.DP
y3h_IT = n3h.IT
y3h_MT = n3h.MT
y3h_V = n3h.V
y3h_R = n3h.R
#Average Ampere
slope3h_A, intercept3h_A, rvalue3h_A, pvalue3h_A, stderr3h_A = linregress(x3h, y3h_A)
processed3h_A = float((slope3h_A) / (head3h_A))
#Intake Pressure
slope3h_IP, intercept, rvalue, pvalue, stderr = linregress(x3h, y3h_IP)
processed3h_IP = float((slope3h_IP) / (head3h_IP))
#Discharge Pressure
slope3h_DP, intercept, rvalue, pvalue, stderr = linregress(x3h, y3h_DP)
processed3h_DP = float((slope3h_DP) / (head3h_DP))
#Intake Temperature
slope3h_IT, intercept, rvalue, pvalue, stderr = linregress(x3h, y3h_IT)
processed3h_IT = float((slope3h_IT) / (head3h_IT))
#Motor Temperature
slope3h_MT, intercept, rvalue, pvalue, stderr = linregress(x3h, y3h_MT)
processed3h_MT = float((slope3h_MT) / (head3h_MT))
#Vibration
slope3h_V, intercept, rvalue, pvalue, stderr = linregress(x3h, y3h_V)
processed3h_V = float((slope3h_V) / (head3h_V))
#Rate
slope3h_R, intercept, rvalue, pvalue, stderr = linregress(x3h, y3h_R)
processed3h_R = float((slope3h_R) / (head3h_R))
#3 hours dataframe
data3h = { 'A':[processed3h_A], 
        'IP':[processed3h_IP], 
        'DP':[processed3h_DP], 
        'IT':[processed3h_IT], 
        'MT':[processed3h_MT], 
        'V':[processed3h_V], 
        'R':[processed3h_R] }
df3h = pd.DataFrame(data3h)
#3 hours prediction
new_prediction3h = logistic_regression.predict(df3h)

#Output
def status(x):
    x = int(x)
    if(x==0):
        return "Running"
    elif(x==1):
        return "Low PI"
    elif(x==2):
        return "Pump Wear"
    elif(x==3):
        return "Tubing Leak"
    elif(x==4):
        return "Higher PI"
    elif(x==5):
        return "Increase in Frequency"
    elif(x==6):
        return "Open Choke"
    elif(x==7):
        return "Increase in Watercut"
    elif(x==8):
        return "Sand Ingestion"
    elif(x==9):
        return "Closed Valve"
    else:
        return "Unidentified"

threedays = status(new_prediction3d[0])
oneday = status(new_prediction1d[0])
threehours = status(new_prediction3h[0])

#Set User Input Data as New Data to the ESP Data   
df_new = input.tail(1).iloc[:,1:].reset_index(drop=True)
a_series = pd.Series(new_list, index = df_new.columns)
df_new = df_new.append(a_series,ignore_index=True)    

#Calculate percentage change
pct = []
for i in df_new.columns:
    pct.append(df_new[i].pct_change()[1])
    
current = status(logistic_regression.predict([pct])[0])    

days = input.TIME.dt.day-4
hour = input.TIME.dt.hour
new_date = []
for i in range(len(days)):
    new_date.append(str(days[i]) + ' day ' + str(hour[i])+ ' hour')

input_new = input.copy()

input_new['TIME'] = new_date


#PLOTTING
fig = px.line(title = 'ESP Data Chart',width=950,height=500)
for i in range(len(input_new.columns[1:])):
    fig.add_scatter(x = input_new['TIME'], y = input_new[input_new.columns[i+1]], name = legend[i],mode='lines+markers') 



#OUTPUT (RIGHT SIDE)
st.markdown('''
Welcome! This program is able to perform real time prediction of ESP pump failure by 
utilizing machine learning (**logistic regression**).

First, you need to insert values to all the input in left side. The result, which are 
problem predictions at different intervals and the ESP data plot, is presented below.
''')

st.info(''' **ESP Sensor Reading**

* Current Status: {}

* Last 3 Hours: {}

* Last Day: {}

* Last 3 Days {}

'''.format(current,threehours,oneday,threedays))

st.plotly_chart(fig)
check_input = st.checkbox('Display ESP Data')
check_df = st.checkbox('Display Dataset')
check_score = st.checkbox('Display Model Accuracy')

if check_input:
    st.write(input_new)
if check_df:
    st.write(data)
if check_score:
    st.info('Model Accuracy: {} '. format(accuracy_score(y_pred,y_test)))


