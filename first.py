

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

test_data = pd.read_csv("/kaggle/input/uomds20/cse_DS_Intro3TEST.csv")
train_data = pd.read_csv("/kaggle/input/uomds20/cse_DS_Intro3TRAIN.csv")



final_train = train_data.loc[:, train_data.columns != "ID"]
X = final_train.loc[:, final_train.columns != "Class"]
Y = final_train.loc[:, final_train.columns == "Class"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.001, random_state=1)

logreg = LogisticRegression(random_state=0)

logreg.fit(X_train, y_train)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
    logreg.score(X_test, np.ravel(y_test, order='C'))))


test_data_pre = test_data.loc[:, test_data.columns != "ID"]
y_pred = logreg.predict(test_data_pre)
with open('/kaggle/working/submission_logistic_reg_1.csv', 'w') as fp:
    fp.write("Id,Prediction\n")
    for i,p in enumerate(y_pred):
        fp.write("{},{}\n".format(i+1, p))

