
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV

dataset=pd.read_csv(r"D:\NIT\JANUARY\9 JAN (Decision Tree)\9th\5. DECESSION TREE CODE\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion= 'gini', max_depth= 4)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)




from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance






import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV

dataset=pd.read_csv(r"D:\NIT\JANUARY\9 JAN (Decision Tree)\9th\5. DECESSION TREE CODE\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion= 'entropy', max_depth= 5)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)




from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance






import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV

dataset=pd.read_csv(r"D:\NIT\JANUARY\9 JAN (Decision Tree)\9th\5. DECESSION TREE CODE\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion= 'entropy', max_depth= 5, min_samples_leaf= 2, min_samples_split= 5)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)




from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance




























































