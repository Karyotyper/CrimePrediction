import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

def convert_to_24(time):
    """str -> str
    converts 12 hours time format to 24 hours
    """
    s = time[:-2] if time[-2:] == "AM" else str(int(time[:2]) + 12) + time[2:8]
    return s[:2]

def set_time_range(x):
    if x == '01' or x=='02' or x=='03' or x=='04' or x=='05':
        return '01-05'

    elif x == '06' or x=='07' or x=='08' or x=='09' or x=='10':
        return '06-10'

    elif x == '11' or x=='12' or x=='13' or x=='14' or x=='15':
        return '11-15'

    elif x == '16' or x=='17' or x=='18' or x=='19' or x=='20':
        return '16-20'

    else:
        return '21-24'

def set_area_range(x):
    if x =='Far North Side' or x=='North Side':
        return 'North'

    elif x =='Central':
        return 'Central'

    elif x=='West Side':
        return 'West'

    else:
        return 'South'


li = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE']
area = ['Southwest Side', 'South Side', 'Far Southwest Side',  'Central', 'West Side',
            'Far North Side', 'North Side']

data = pd.read_csv('final_crime_data.csv')
data = data[data['Primary_Type'].isin(li)]
data = data[data['Area_Name'].isin(area)]
# convert time to 24 hours
data['time'] = data['time'].apply(lambda x: convert_to_24(x))

data['time_range'] = data['time'].apply(lambda x: set_time_range(x))

data['Area_Range'] = data['Area_Name'].apply(lambda x: set_area_range(x))


import numpy as np

msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]

# Convert crime labels to numbers
le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(train.Primary_Type)

# Get binarized weekdays, districts, and hours.
days = pd.get_dummies(train.day)
area_name = pd.get_dummies(train.Area_Range)
hour = pd.get_dummies(train.time_range)
# Build new array
train_data = pd.concat([area_name, hour,days], axis=1)
train_data['Primary_Type'] = crime

training, validation = train_test_split(train_data, train_size=.80)

features = ['Central','West', 'North', 'South']


features = data.time_range.unique().tolist()+features

#implement BernoulliNB
# model = BernoulliNB()
# model.fit(training[features], training['Primary_Type'])
# predicted = np.array(model.predict_proba(validation[features]))

#Logistic Regression for comparison
# model = LogisticRegression(C=.01)
# model.fit(training[features], training['Primary_Type'])
# predicted = np.array(model.predict_proba(validation[features]))

#using decision tree
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(training[features], training['Primary_Type'])
predicted = np.array(model.predict_proba(validation[features]))


print log_loss(validation['Primary_Type'], predicted)


days = pd.get_dummies(test.day)
area_name = pd.get_dummies(test.Area_Range)
hour = pd.get_dummies(test.time_range)
test_data = pd.concat([area_name, hour, days], axis=1)

predicted_val = model.predict(test_data[features])
predicted_val = le_crime.inverse_transform(predicted_val)
result = pd.DataFrame(predicted_val,columns=['Primary_Type'])
l1 =  test['Primary_Type'].tolist()
l2 =  result.Primary_Type.tolist()

from sklearn.metrics import accuracy_score, log_loss

print accuracy_score(l1, l2)
# result.to_csv('results.csv', index=False)
