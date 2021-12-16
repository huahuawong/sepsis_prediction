import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utility_visualisation import missing_values_barplot, remove_plot_ticks
from utility import TimeSeriesDataset, load_pickle, torch_ffill, shock_index, partial_sofa, RollingStatistic
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold

data_dir = './input'

# 1. Loading the dataset and perform quick inspection of data
df = load_pickle(data_dir + '/raw/df.pickle')
labels_binary = load_pickle(data_dir + '/processed/labels/binary.pickle')
labels_utility = load_pickle(data_dir + '/processed/labels/utility_scores.pickle')

df.head(5)
df.info()

missing_values_barplot(df, missing=False)


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plotting random patient's data
# Get person and time
person = df[df['id'] == 777]
tt = range(person.shape[0])

# Plot HR and
ax[0].scatter(tt, person['HR'])
ax[1].scatter(tt, person['Calcium'])

# Plot info
ax[0].set_title('Heart Rate', fontsize=18)
ax[0].set_xlabel('Time', fontsize=16)
ax[0].set_ylabel('HR (bpm)', fontsize=16)

ax[1].set_title('Calcium', fontsize=18)
ax[1].set_xlabel('Time', fontsize=16)
ax[1].set_ylabel('Calcium', fontsize=16)


# Converting current data to ragged data. the data into a tensor format by extending everyones time-series to the same
# size as the maximum length time-series in the data filling in all values past the final time as nan.
#
# The structure keeps track of the original lengths, so we can always convert back, and also keeps track of the columns
# , so we can apply operations only on certain features. This is also going to be very important.

# Load the dataset
dataset = TimeSeriesDataset().load(data_dir + '/raw/data.tsd')
dataset.data.size()

# 2. Preprocessing Data and Extract Features
dataset.data = torch_ffill(dataset.data)
dataset.data[0,:,0]   # check forward fill

dataset['ShockIndex'] = shock_index(dataset)
dataset['PartialSOFA'] = partial_sofa(dataset)

dataset['MaxShockIndex'] = RollingStatistic(statistic='max', window_length=5).transform(dataset['SBP'])
dataset['MinHR'] = RollingStatistic(statistic='min', window_length=8).transform(dataset['HR'])

# Training Model
# Get ML form of the data
X = dataset.to_ml()
y = labels_binary

assert len(X) == len(y)    # Sanity check

# Fill the nans
X[torch.isnan(X)] = -1000

# Choose cross val method
cv = list(KFold(5).split(X))

# Train and predict
clf = RandomForestClassifier()
probas = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')

from sklearn.metrics import accuracy_score, roc_auc_score

preds = (probas[:, 1] > 0.5).astype(int)
acc = accuracy_score(y, preds)
auc = roc_auc_score(y, probas[:, 1])

print('Accuracy: {:.2f}% \nAUC: {:.3f}'.format(round(acc * 100, 2), auc))
