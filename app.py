from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("ForbesAmericasTopColleges2019.csv",header=None, skiprows=1, names=['Rank', 'Name', 'City', 'State', 'category',
       'Undergraduate_Population', 'Student_Population', 'Net Price',
       'Average Grant Aid', 'Total_Annual_Cost', 'Alumni Salary',
       'Acceptance Rate', 'SAT Lower', 'SAT Upper', 'ACT Lower', 'ACT Upper',
       'Website'])

df1 = df.loc[0:,['Rank', 'Name', 'City', 'State', 'category','Student_Population','Total_Annual_Cost']]
df1["Rank"] = df1["Rank"].astype("int")

cat_cols = df1.select_dtypes(["O"]).keys()

for var in cat_cols:
    df1[var].fillna(df1[var].mode()[0], inplace=True)

df2 = df1["Name"]
df2 = pd.DataFrame({"Name":df2})
df3 = df1["City"]
df3 = pd.DataFrame({"City":df3})
df4 = df1["State"]
df4 = pd.DataFrame({"State":df4})

value = df1["Name"].value_counts().keys()
value7 = df1["Name"].value_counts().keys()
value1 = df1["City"].value_counts().keys()
value2 = df1["State"].value_counts().keys()
value3 = df1["category"].value_counts().keys()

for num,var in enumerate(value):
    num+=1
    df1["Name"].replace(var, num, inplace=True)

for num, var in enumerate(value1):
    num+=1
    df1["City"].replace(var, num, inplace=True)

for num,var in enumerate(value2):
    num+=1
    df1["State"].replace(var, num, inplace=True)

for num,var in enumerate(value3):
    num+=1
    df1["category"].replace(var, num, inplace=True)

X = df1.drop("Rank", axis=1)
y = df1["Rank"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

sc=StandardScaler()
sc.fit(X_train,y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


model = joblib.load("country_rank_prediction.pkl")

def country_rank_prediction(model,Name, City, State, category, Student_Population, Total_Annual_Cost):
    for num,var in enumerate(value):
        if var == Name:
            Name = num
    for num,var in enumerate(value1):
        if var == City:
            City = num
    for num,var in enumerate(value2):
        if var == State:
            State = num
    for num,var in enumerate(value3):
        if var == category:
            category = num
    x = np.zeros(len(X.columns))
    x[0] = Name
    x[1] = City
    x[2] = State
    x[3] = category
    x[4] = Student_Population
    x[5] = Total_Annual_Cost
    
    x = sc.transform([x])[0]
    return model.predict([x])[0]

app=Flask(__name__)

@app.route("/")
def home():
    value7 = list(df2["Name"].value_counts().keys())
    value7.sort()
    value10 = list(df3["City"].value_counts().keys())
    value10.sort()
    value11 = list(df4["State"].value_counts().keys())
    value11.sort()
    return render_template("index.html",value=value7, value01=value10,value02=value11)

@app.route("/predict", methods=["POST"])
def predict():
    Name = request.form["Name"]
    City = request.form["City"]
    State = request.form["State"]
    category = request.form["category"]
    Student_Population = request.form["Student_Population"]
    Total_Annual_Cost = request.form["Total_Annual_Cost"]
    
    predicated_price = country_rank_prediction(model,Name, City, State, category, Student_Population, Total_Annual_Cost)
    predicated_price = predicated_price.astype("int")

    return render_template("index.html", prediction_text="Country Rank of:- {}".format(predicated_price))


if __name__ == "__main__":
    app.run()    
    