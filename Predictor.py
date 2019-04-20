import pandas as pd
from sklearn.externals import joblib
from flask import Flask,render_template,request
app = Flask(__name__)


s1 = pd.read_csv('dataset.csv')
s2 = pd.read_csv('datasetrav.csv')
s3 = pd.read_csv('datasetravnew.csv')

df = pd.concat([s1,s2,s3])
main_category_set = sorted(list(set(df['main_category'].values)))
category_set = sorted(list(set(df['category'].values)))


classifier = joblib.load('PSKC_model.sav')
vectorizer = joblib.load('vectorizer.sav')
labelEncoder1 = joblib.load('labelEncoder1.sav')
labelEncoder2 = joblib.load('labelEncoder2.sav')
onehotencoder = joblib.load('onehotencoder.sav')
sc = joblib.load('sc.sav')

onehotencoder.feature_indices = onehotencoder.feature_indices_
onehotencoder.active_features = onehotencoder.active_features_
onehotencoder.n_values = onehotencoder.n_values_
onehotencoder._n_values = onehotencoder.n_values
onehotencoder._legacy_mode = True
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

#
# pred_label = prediction('Art', 'Art', 10, 9000.0,"A 3D photo-based artist's book exploring a fictional subterranean labyrinth")
# print(pred_label)
@app.route('/')
def index():
     return render_template('register.html', mc = main_category_set,c=category_set)

@app.route('/pskc',methods=['GET','POST'])
def send():
    age='a'
    if request.method=='POST':
        category=request.form['category']
        main_category = request.form['main_category']
        start_date = request.form['sdate']
        end_date=request.form['edate']
        goal = request.form['goal']
        description = request.form['description']
        from datetime import date
        sdarr = start_date.split('-')
        edarr = end_date.split('-')
        f_date = date(int(sdarr[0]), int(sdarr[1]), int(sdarr[2]))
        l_date = date(int(edarr[0]), int(edarr[1]), int(edarr[2]))
        delta = l_date - f_date
        days = delta
        pred_label = prediction(category, main_category, days, goal,description)
        return render_template('f1.html',pred = pred_label)
    return render_template('f1.html')


if __name__ =="__main__":
    app.run(debug=True)
