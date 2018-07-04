import pandas as pd


# prepare dataset

event_data_2016 = pd.read_csv("snippet_2016.csv")
event_data_2017 = pd.read_csv("snippet_2017.csv")
event_data = event_data_2016.append(event_data_2017)
print event_data.Primary_Type.unique.tolist()
#plot all crime type counts
# import seaborn as sns
# sns.set(font_scale=0.75)
# sns.set(style='darkgrid')
# import matplotlib.pyplot as plt
# g = sns.factorplot("Primary_Type", data=event_data, aspect=1.5,
#                    kind="count", color="b",
#                    order=event_data['Primary_Type'].value_counts().index)
# g.set_xticklabels(rotation=90)
# plt.show()


# chosen crime category after analysing the graph and values
crimes_chosen = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'ASSAULT', 'DECEPTIVE PRACTICE']
#
event_data = event_data[event_data['Primary_Type'].isin(crimes_chosen)]


#incorporate community area Name
area_data = pd.read_csv('community.csv')
event_data = pd.merge(event_data, area_data, how='left', on='Community Area')
del event_data['Community Area']


event_data['date'], event_data['time'] = zip(*event_data['Date'].apply(lambda x: x.split(' ', 1)))


event_data['date'] = pd.to_datetime(event_data['date'])
event_data['day'] = event_data['date'].dt.weekday_name


#plot area vs crime plot
import seaborn as sns
sns.set(font_scale=0.75)
sns.set(style='darkgrid')
import matplotlib.pyplot as plt
g = sns.factorplot("day", data=event_data, aspect=1.5,
                   kind="count", color="b",
                   order=event_data['day'].value_counts().index)
g.set_xticklabels(rotation=90)
plt.show()

print len(event_data.time.unique())

event_data.to_csv('final_crime_data.csv')
