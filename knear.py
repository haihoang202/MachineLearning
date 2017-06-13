import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import pickle

df = pd.read_csv('breast-cancer-wisconsin.txt');
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'],1))
print(x)
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size= 0.2)

clf = neighbors.KNeighborsClassifier()
# clf.fit(x_train, y_train)

# Save to pickle
# with open('knear.pickle','wb') as f:
#     pickle.dump(clf, f)

# Load pickle
pickle_in = open ('knear.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
num = len(example_measures)
print(num)
example_measures = example_measures.reshape(num,-1)

prediction = clf.predict(example_measures)

print(prediction)
