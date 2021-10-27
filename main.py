from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split


data = load_iris(as_frame=True)# load as dataframe
x_data = data.data.to_numpy()
y_data = data.target.values


X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=666)

model = RandomForestClassifier()
model.fit(X_train,y_train)
score = model.score(X_test,y_test)
print('test acc = {0}'.format(score))
