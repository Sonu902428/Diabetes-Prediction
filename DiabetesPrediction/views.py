import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')
def result(request):
    df = pd.read_csv(r"D:\Diabetes Prediction\DiabetesPrediction\diabetes.csv")

    x_train,x_test,y_train,y_test = train_test_split(df.drop(columns=['Outcome']),df['Outcome'],test_size=0.2)
    lb = LogisticRegression()
    lb.fit(x_train,y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    
    predict = lb.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])

    output = ""
    if predict == [1]:
        output = "Positive"

    elif predict == [0]:
        output = "Negative"

    return render(request, "predict.html", {"Output":output})