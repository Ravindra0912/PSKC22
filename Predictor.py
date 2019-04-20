import pandas as pd
from sklearn.externals import joblib

classifier = joblib.load('PSKC_model.sav')
vectorizer = joblib.load('vectorizer.sav')
labelEncoder1 = joblib.load('labelEncoder1.sav')
labelEncoder2 = joblib.load('labelEncoder2.sav')
onehotencoder = joblib.load('onehotencoder.sav')
onehotencoder = joblib.load('onehotencoder.sav')
sc = joblib.load('sc.sav')


# User input
def prediction(category, main_category, days, goal, description):
    user_input = []
    user_input.append([category, main_category, days, goal, description])

    tup = pd.DataFrame(user_input)
    tup_desc = tup.iloc[:, -1]

    tup_vector = vectorizer.transform(tup_desc)
    tup_tfidf = pd.DataFrame(tup_vector.toarray())

    tup_attr = tup.iloc[:, :-1]
    tup_attr.iloc[:, 0] = labelEncoder1.transform(tup_attr.iloc[:, 0])
    tup_attr.iloc[:, 1] = labelEncoder2.transform(tup_attr.iloc[:, 1])

    tup_attr = onehotencoder.transform(tup_attr).toarray()

    l = [pd.DataFrame(tup_attr), tup_tfidf]
    tup_newd = pd.concat(l, axis=1)
    X = tup_newd.iloc[:, :].values

    X = sc.transform(X)

    tup_pred = classifier.predict(X)
    prediction = tup_pred[0]
    return prediction


pred_label = prediction('Art', 'Art', 10, 9000.0,"A 3D photo-based artist's book exploring a fictional subterranean labyrinth")
print(pred_label)