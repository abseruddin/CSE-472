import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import math


def readTelcoData(data):
  cols = data.columns

  # find out column name with missing values
  missingColumns=[]
  for col in cols:
    if " " in np.unique(data[col]):
      missingColumns.append(col)

  data["TotalCharges"].replace({" ": "NaN"}, inplace=True)

  # changing data types
  convert_dict = {"TotalCharges": float,
                  }
  data = data.astype(convert_dict)

  # handling missing data
  data["TotalCharges"].fillna(data["TotalCharges"].mean())

  # dropping columns
  if "customerID" in cols:
    data.drop(["customerID"], axis = 1, inplace = True)

  # print(data.dtypes)
  # feature scaling
  featureScalingCols = ["tenure","MonthlyCharges","TotalCharges"]

  min_max_scaler = preprocessing.MinMaxScaler()
  data[featureScalingCols] = min_max_scaler.fit_transform(data[featureScalingCols])
  
  # one hot encoding catagorical data
  if 'gender' in cols:
    data['gender'].replace({"Male":1,"Female":0},inplace = True)
  if 'Partner' in cols:
    data['Partner'].replace({"No":0,"Yes":1},inplace = True)
  if 'Dependents' in cols:
    data['Dependents'].replace({"No":0,"Yes":1},inplace = True)
  if 'PhoneService' in cols:
    data['PhoneService'].replace({"No":0,"Yes":1},inplace = True)
  if 'PaperlessBilling' in cols:
    data['PaperlessBilling'].replace({"No":0,"Yes":1},inplace = True)
  if 'MultipleLines' in cols:
    data['MultipleLines'].replace({"No phone service":0,"No":1,"Yes":2},inplace = True)
  if 'InternetService' in cols:
    data = pd.get_dummies(data, columns=["InternetService"], prefix="internet")
  someCols = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract"]
  prefixes = ["security","backup","protection","tech","tv","movies","con"]
  i = 0
  for col in someCols:
    if col in cols:
      data = pd.get_dummies(data, columns=[col], prefix=prefixes[i])
      i = i + 1
  if 'PaymentMethod' in cols:
    data = pd.get_dummies(data, columns=["PaymentMethod"], prefix="Pay")
  data['Churn'].replace({"No":-1,"Yes":1},inplace = True)
  # print(data.dtypes)
  # x = data.iloc[:,data.columns!="Churn"].values
  # y = data.iloc[:,data.columns=="Churn"].values
  # outputColumn = data.iloc[:,data.columns=="Churn"]
  # data.drop(["Churn"], axis = 1, inplace = True)
  cols = [col for col in data.columns if col != 'Churn'] + ['Churn']
  data = data.reindex(columns=cols)
  x = data.drop(columns=data.columns[-1])
  y = data[data.columns[-1]]
  return x,y,data

def readAdultData(data):
  cols = data.columns

  # find out column name with missing values
  imp = SimpleImputer(missing_values=" ?", strategy='most_frequent')
  data = pd.DataFrame(imp.fit_transform(data),
                    columns=data.columns,
                    index=data.index)

  # changing data types
  convert_dict = {"age": int,
                  "fnlwgt": int,
                  "education-num": int,
                  "capital-gain": int,
                  "capital-loss": int,
                  "hours-per-week": int,
                  }
  data = data.astype(convert_dict)

  # feature scaling
  featureScalingCols = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
  min_max_scaler = preprocessing.MinMaxScaler()
  data[featureScalingCols] = min_max_scaler.fit_transform(data[featureScalingCols])

  # one hot encoding catagorical data
  if 'sex' in cols:
    data['sex'].replace({" Male":1," Female":0},inplace = True)
  if 'income' in cols:
    data['income'].replace({" <=50K":1," >50K":0},inplace = True)
  someCols = ["workclass","education","marital-status","occupation","relationship","race","native-country"]
  prefixes = ["work","edu","marry","occu","relation","rac","country"]
  i = 0
  for col in someCols:
    if col in cols:
      data = pd.get_dummies(data, columns=[col], prefix=prefixes[i])
      i = i + 1
  # for col in data.columns:
  #   print(col)
  #   print(np.unique(data[col]))
  # x = data.iloc[:,data.columns!="income"].values
  # y = data.iloc[:,data.columns=="income"].values
  cols = [col for col in data.columns if col != 'income'] + ['income']
  data = data.reindex(columns=cols)
  x = data.drop(columns=data.columns[-1])
  y = data[data.columns[-1]]
  return x,y,data
  pass

def readCreditData(data):
  featureScalingCols = [col for col in data.columns]
  min_max_scaler = preprocessing.MinMaxScaler()
  data[featureScalingCols] = min_max_scaler.fit_transform(data[featureScalingCols])

  x = data.iloc[:,data.columns!="Class"].values
  y = data.iloc[:,data.columns=="Class"].values
  cols = [col for col in data.columns if col != 'Class'] + ['Class']
  data = data.reindex(columns=cols)
  return x,y,data
  pass

def getDataset(self,data):
    x = data.drop(columns=data.columns[-1])
    y = data[data.columns[-1]]
    y = y.to_numpy().reshape(x.shape[0],1)
    intercept = np.ones((x.shape[0],1))
    x = np.concatenate((intercept, x), axis=1)
    data = pd.concat([x,y],axis=1)
    return x,y,data

class Regression:
  def __init__(self,noOfFeature,learningRate,threshold):
    self.noOfFeature = noOfFeature
    self.learningRate = learningRate
    self.threshold = threshold
    self.w = np.zeros(self.noOfFeature)
  def calculateH(self,z):
    result=np.tanh(z)
    return result
  def run(self,x,y):
    intercept = np.ones((x.shape[0],1))
    x = np.concatenate((intercept, x), axis=1)
    n,m=x.shape
    self.w = np.zeros((m,1))
    for j in range(700):
      h = np.tanh(np.dot(x,self.w))
      print(h)
      loss = 0
      for i in range(len(h)):
        e = y[i] - h[i]
        loss = loss + e*e
      loss = loss/(len(h))
      # loss = np.sum((y-h)**2)/n
      # print(loss)
      
      
      q = 1 - (h ** 2)
      p = (y-h)*q
      gradient=(np.dot(x.T,p))/len(y)
      # print(gradient)
      #gradient = np.dot(np.dot(p,q),x)#np.dot((y-h)*h*(1-h*h),x.T)
      
      self.w = self.w + np.multiply(self.learningRate, gradient)

  def predict( self, x ) :
    intercept = np.ones((x.shape[0],1))
    x = np.concatenate((intercept, x), axis=1)
    h = self.calculateH(np.dot(x,self.w))    
           
    y = np.where( h > 0, 1, -1 )        
    return y
 
class Adaboost:
  def __init__(self,examples,Lweak,k):
    self.totalSample = len(examples)
    self.k = k
    self.w = np.full(self.totalSample,1/self.totalSample)
    self.examples = examples
    self.Lweak = Lweak
    pass

  def resample(self):
    newIndex = np.random.choice(np.arange(0,self.totalSample),self.totalSample,p = self.w)
    newExamples = []
    a=self.examples.values
    for i in newIndex:
      newExamples.append(a[i])
    newExamples=pd.DataFrame(newExamples,columns=self.examples.columns)
    return newExamples
    pass

  def normalize(self):
    s = np.sum(self.w)
    for i in range(len(self.w)):
      self.w[i] = self.w[i]/s
    pass
  
  def getDataset(self,data):
    x = data.drop(columns=data.columns[-1])
    y = data[data.columns[-1]]
    data = pd.concat([x,y],axis=1)
    y = y.to_numpy().reshape(x.shape[0],1)
    # intercept = np.ones((x.shape[0],1))
    # x = np.concatenate((intercept, x), axis=1)
    
    return x,y,data
  def adaboost(self):
    z = []
    h = []
    X=self.examples.iloc[:,:-1].values
    Y=self.examples.iloc[:,-1].values
    # print("B")
    for i in range(self.k):
      data = self.resample()
      x,y,zw=self.getDataset(data)
     
      #x = data.iloc[:,:-1].values
      #y = data.iloc[:,-1].values
      self.Lweak.run(x,y)
      # print("A")
      error = 0
      predictions = self.Lweak.predict(X)
      for j in range(self.totalSample):
        if predictions[j] != Y[j]:
          error = error + self.w[j]
      if error > 0.5:
        continue
      for j in range(self.totalSample):
        if  predictions[j]== Y[j]:
          self.w[j] = self.w[j] *(error/(1-error))
      self.normalize()
      z.append(np.log2((1-error)/error))
      h.append(self.Lweak.w)
      return h,z
      # print(z)
      
  def findWeightedMajority(self,h,z,X):
    n,m=X.shape
    weightedSum=np.zeros((n,1))
    for i in range(len(h)):
        yPred=np.tanh(np.dot(X,h[i]))#self.Lweak.predict(np.dot(X,h[i]))
        weightedValue=np.multiply(z[i],yPred)
        weightedSum+=weightedValue
    return weightedSum

  

def entrophy(col):
  uniqueLabels = np.unique(col)
  count = np.full(len(uniqueLabels),0)
  i = 0
  for label in uniqueLabels:
    for data in col:
      if data == label:
        count[i] = count[i] + 1
    i = i+1
  total = sum(count)
  b = 0
  for i in range(len(uniqueLabels)):
    b = b + (count[i]/total)*math.log2(count[i]/total)
  b = b*-1
  return b

def informationGain(data,featureName,parentName):
  totalEntrophy = entrophy(data[parentName])
  uniqueLabels = np.unique(data[featureName])
  e = 0
  for label in uniqueLabels:
    df1 = data[data[featureName]==label]
    e = e + entrophy(df1[parentName])*len(df1)/len(data)
  # featureIndex = data.columns.get_loc(featureName)
  # print(data[data[:,featureIndex]==0,-1])
  gain = totalEntrophy - e
  return gain
  pass

def mainforTelco():
    # totalFeature = input("Enter number of feature ")
    # totalFeature = int(totalFeature)
    data = pd.read_csv('./telcodata.csv')
    cols = data.columns
    totalFeature = len(cols)
    gains = []
    seperatedColNames = []
    for col in cols:
        gains.append(informationGain(data,col,'Churn'))
    indexes = sorted(range(len(gains)), key=lambda i: gains[i])[-totalFeature:]
    for index in indexes:
        seperatedColNames.append(cols[index])
    seperateData = data.loc[:, data.columns.isin(seperatedColNames)]
    x,y,data = readTelcoData(seperateData)
    return x,y,data

def mainforAdult():
    # totalFeature = input("Enter number of feature ")
    # totalFeature = int(totalFeature)
    data = pd.read_csv('./adultdata.csv')
    cols = data.columns
    totalFeature = len(cols)
    gains = []
    seperatedColNames = []
    for col in cols:
        gains.append(informationGain(data,col,'income'))
    indexes = sorted(range(len(gains)), key=lambda i: gains[i])[-totalFeature:]
    for index in indexes:
        seperatedColNames.append(cols[index])
    seperateData = data.loc[:, data.columns.isin(seperatedColNames)]
    x,y,data = readAdultData(seperateData)
    return x,y,data

def mainforCredit():
    totalFeature = input("Enter number of feature ")
    totalFeature = int(totalFeature)
    data = pd.read_csv('./creditCard.csv')
    cols = data.columns
    totalFeature = len(cols)
    gains = []
    seperatedColNames = []
    t = -totalFeature
    for col in cols:
        gains.append(informationGain(data,col,'Class'))
    indexes = sorted(range(len(gains)), key=lambda i: gains[i])[t:]
    for index in indexes:
        seperatedColNames.append(cols[index])
    seperateData = data.loc[:, data.columns.isin(seperatedColNames)]
    x,y,data = readCreditData(seperateData)
    return x,y,data

def final(x,y,data):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=4651)
    # intercept = np.ones((train_x.shape[0],1))
    # train_x = np.concatenate((intercept, train_x), axis=1)

    intercept = np.ones((test_x.shape[0],1))
    test_x = np.concatenate((intercept, test_x), axis=1)

    regression = Regression(5,.01,.6)
    ada = Adaboost(data,regression,5)
    h,z = ada.adaboost()
    weightedSum = ada.findWeightedMajority(h,z,train_x)

    output = y = np.where( weightedSum > 0, 1, -1 )
    outputs = confusion_matrix(train_y,y_pred)
    tn,fp,fn,tp=outputs.ravel()
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print("Accuracy: ",accuracy)
def forRegression(x,y,data):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=4651)
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    regression = Regression(5,.01,.6)
    regression.run(train_x,train_y)
    y_pred = regression.predict(train_x)
    i = 0
    error = 0
    print(y_pred.shape,train_y.shape)
    for i in range(np.size(y_pred)):
      if train_y[i]!=y_pred[i]:
        error = error + 1
    print(error/np.size(y_pred))
    outputs = confusion_matrix(train_y,y_pred)
    tn,fp,fn,tp=outputs.ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    falseRate = fp/(fp+tp)
    f1 = 2*tp/(2*tp+fp+fn)
    print("sensitivity: ",sensitivity)
    print("specificity: ",specificity)
    print("precision: ",precision)
    print("false discovery Rate",falseRate)
    print("f1 score: ",f1)
def main():
    dataset = input("Dataset No ")
    dataset = int(dataset)
    
    if dataset==1:
        x,y,data = mainforTelco()
        forRegression(x,y,data)   
        #final(x,y,data)
    elif dataset==2:
        x,y,data = mainforAdult()
        final(x,y,data)
    else:
        x,y,data = mainforCredit()
        final(x,y,data)
    
if __name__ == "__main__":
    main()
    pass

