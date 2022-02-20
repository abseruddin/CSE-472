import numpy as np
from sklearn.preprocessing import OneHotEncoder
import math
import pandas as pd
from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import train_test_split
import pickle
import os
from sklearn.metrics import accuracy_score,f1_score,classification_report

oneHotEnc = OneHotEncoder(sparse=False)

   
def readMnistData():
    X1,y1 = loadlocal_mnist(images_path='train-images.idx3-ubyte',labels_path='train-labels.idx1-ubyte')
    X2,y2 = loadlocal_mnist(images_path='t10k-images.idx3-ubyte',labels_path='t10k-labels.idx1-ubyte')


    train_X = np.reshape(X1,(X1.shape[0],28,28,1))
    train_Y = np.reshape(y1,(y1.shape[0],1))
    train_X = train_X/255
    train_Y_encoded = oneHotEnc.fit_transform(train_Y)

    test_ratio = 0.5
    test_X,validation_X,test_Y,validation_Y = train_test_split(X2,y2,test_size = test_ratio)

    validation_X = np.reshape(validation_X,(validation_X.shape[0],28,28,1))
    validation_X = validation_X/255
    validation_Y = np.reshape(validation_Y,(validation_Y.shape[0],1))
    validation_Y_encoded = oneHotEnc.fit_transform(validation_Y)

    test_X = np.reshape(test_X,(test_X.shape[0],28,28,1))
    test_X = test_X/255
    test_Y = np.reshape(test_Y,(test_Y.shape[0],1))
    test_Y_encoded = oneHotEnc.fit_transform(test_Y)
    
    return train_X,train_Y,train_Y_encoded,test_X,test_Y,test_Y_encoded,validation_X,validation_Y,validation_Y_encoded
 
def loadSinglebatch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def loadCifarData():
    """ load all of cifar """
    singleX = []
    singleY = []
    for b in range(1,6):
        f = os.path.join('./cifar-10-python/cifar-10-batches-py/', 'data_batch_%d' % (b, ))
        X, Y = loadSinglebatch(f)
        singleX.append(X)
        singleY.append(Y)
    trainX = np.concatenate(singleX)
    trainY = np.concatenate(singleY)
    
    testX, testY = loadSinglebatch(os.path.join('./cifar-10-python/cifar-10-batches-py/', 'test_batch'))
    return trainX, trainY, testX, testY

def readCifarData():
    # Load the raw CIFAR-10 data
    # cifar10_dir = './cifar-10-python/cifar-10-batches-py/'
    X1,y1,X2,y2 = loadCifarData()

    train_X = np.reshape(X1,(X1.shape[0],32,32,3))
    train_Y = np.reshape(y1,(y1.shape[0],1))
    train_X = train_X/255
    train_Y_encoded = oneHotEnc.fit_transform(train_Y)

    test_ratio = 0.5
    test_X,validation_X,test_Y,validation_Y = train_test_split(X2,y2,test_size = test_ratio)

    # Subsample the data
    # mask = range(num_training, num_training + num_validation)
    # X_val = X_train[mask]
    # y_val = y_train[mask]
    # mask = range(num_training)
    # X_train = X_train[mask]
    # y_train = y_train[mask]
    # mask = range(num_test)
    # X_test = X_test[mask]
    # y_test = y_test[mask]

    # x_train = X_train.astype('float32')
    # x_test = X_test.astype('float32')
    validation_X = np.reshape(validation_X,(validation_X.shape[0],32,32,3))
    validation_X = validation_X/255
    validation_Y = np.reshape(validation_Y,(validation_Y.shape[0],1))
    validation_Y_encoded = oneHotEnc.fit_transform(validation_Y)

    test_X = np.reshape(test_X,(test_X.shape[0],32,32,3))
    test_X = test_X/255
    test_Y = np.reshape(test_Y,(test_Y.shape[0],1))
    test_Y_encoded = oneHotEnc.fit_transform(test_Y)

    return train_X,train_Y,train_Y_encoded,test_X,test_Y,test_Y_encoded,validation_X,validation_Y,validation_Y_encoded
  
def padding(a,p):
    b = np.zeros((a.shape[0],a.shape[1]+2*p,a.shape[2]+2*p,a.shape[3]))
    b[:,p:a.shape[1]+p,p:a.shape[2]+p,:] = a
    return b
    pass

class Conv:
    def __init__(self,n,f,s,p,learningRate):
        self.n = n
        self.f = f
        self.s = s
        self.p = p
        self.learningRate = learningRate
        self.weights = []
        self.bias = []
        self.dw = []
        self.db = []
        self.da = []
        self.dim1 = 0
        self.dim2 = 0
        self.a = []
        self.batchSize = 1


    def initializeWeight(self,prevSize,he):
        self.weights = np.random.randn(self.f,self.f,prevSize,self.n)*np.sqrt(2/he)
        self.bias = np.zeros(self.n) # np.random.randn(self.n)
        self.dw = np.zeros(self.weights.shape)
        self.db = np.zeros(self.bias.shape)
        
        pass

        
    def forward(self,a1):

        self.batchSize = a1.shape[0]
        self.dim1 = int((a1.shape[1] + 2*self.p - self.f)/self.s + 1)
        self.dim2 = int((a1.shape[2] + 2*self.p - self.f)/self.s + 1)
        
        result = np.zeros((self.batchSize,self.dim1,self.dim2,self.n))

        self.da = np.zeros(a1.shape)
        a1 = padding(a1,self.p)
        self.a = a1

        for l in range(self.batchSize):
            for i in range(self.dim1):
                d1 = i*self.s
                for j in range(self.dim2):
                    d2 = j*self.s
                    for k in range(self.n):
                        r = np.multiply(a1[l,d1:d1+self.f,d2:d2+self.f,:],self.weights[:,:,:,k])
                        result[l][i][j][k] = np.sum(r)+self.bias[k]
                    
        return result
        pass
    
    
    def backward(self,result):

        self.dw = np.zeros(self.weights.shape)
        self.db = np.zeros(self.bias.shape)
        da1 = np.zeros(self.a.shape)

        for l in range(self.batchSize):
            for i in range(self.dim1):
                d1 = i*self.s
                for j in range(self.dim2):
                    d2 = j*self.s
                    for k in range(self.n):
                        da1[l,d1:d1+self.f,d2:d2+self.f,:] += self.weights[:,:,:,k]*result[l,i,j,k]
                        self.dw[:,:,:,k]+=self.a[l,d1:d1+self.f,d2:d2+self.f,:]*result[l,i,j,k]
                        self.db[k]+=result[l,i,j,k]
        if self.p!=0:
            self.da = da1[:,self.p:-1*self.p,self.p:-1*self.p,:]
        else:
            self.da = da1
        self.updateWeight()
        return self.da
        pass

    def updateWeight(self):
        # print(self.dw.shape,self.weights.shape)
        # print(self.bias.shape,self.db.shape)
        # print(self.weights.flatten()[:5])
        # print(self.bias.flatten()[:5])
        self.weights = self.weights - self.learningRate*self.dw
        self.bias = self.bias - self.learningRate * self.db
        # print(self.weights.flatten()[:5])
        # print(self.bias.flatten()[:5])
        pass

# np.random.seed(1)
# A_prev = np.random.randn(10,5,7,4)
# W = np.random.randn(3,3,4,8)
# b = np.random.randn(1,1,1,8)
# c = Conv(8,3,2,1,0.01)
# Z = []
# Z = c.forward(A_prev,W,b[0,0,0,:])
# for i in range(10):
#     Z.append(c.forward(A_prev[i],W,b[0,0,0,:]))
# print("Z's mean =\n", np.mean(Z))
# print("Z[3,2,1] =\n", Z[3][2][1])
# print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])


class Pooling:
    def __init__(self,f,s):
        self.f = f
        self.s = s
        self.dim1 = 0
        self.dim2 = 0
        self.a = []
        self.da = []
        self.batchSize = 1
        pass
    def initializeWeight(self):
        pass

    
    def forward(self,a1):

        self.batchSize = a1.shape[0]

        self.dim1 = int((a1.shape[1] - self.f)/self.s + 1)
        self.dim2 = int((a1.shape[2] - self.f)/self.s + 1)

        self.a = a1
        self.da = np.zeros(a1.shape)

        result = np.zeros((self.batchSize,self.dim1,self.dim2,a1.shape[3]))
        
        for l in range(self.batchSize):
            for i in range(self.dim1):
                d1 = i*self.s
                for j in range(self.dim2):
                    d2 = j*self.s
                    for k in range(a1.shape[3]):
                        result[l][i][j][k] = np.max(a1[l,d1:d1+self.f,d2:d2+self.f,k])

        return result
        pass

    
    def backward(self,result):
        self.da = np.zeros(self.a.shape)
        for l in range(self.batchSize):
            for i in range(self.dim1):
                d1 = i*self.s
                for j in range(self.dim2):
                    d2 = j*self.s
                    for k in range(result.shape[3]):
                        m = np.argmax(self.a[l,d1:d1+self.f,d2:d2+self.f,k])
                        r = int(d1+int(m/self.f))
                        c = int(d2+m%self.f)
                        self.da[l][r][c][k] += result[l][i][j][k]
                    # m = np.max(self.a[d1:d1+self.f,d2:d2+self.f,k])
                    # mask = (self.a[d1:d1+self.f,d2:d2+self.f,k]==m)
                    # self.da[d1:d1+self.f,d2:d2+self.f,k] += mask*result[i,j,k]
        
        return self.da
        pass
    def updateWeight(self,dw,db):
        pass


# np.random.seed(1)
# A = np.random.randn(5,5,3,2)
# p = Pooling(2,1)
# r = []
# dA = np.random.randn(5,4,2,2)
# p.forward(A)
# r = p.backward(dA)
# for i in range(5):
#     a = p.forward(A[i])
#     r.append(p.backward(dA[i]))
# print("mean = ",np.mean(dA))
# print(r[1][1])

# a = np.zeros((4,5,5))
# a[1][2][3] = 5
# a[1][3][2] = 4
# a[1][3][3] = 7
# m=np.argmax(a[1,2:4,2:4])
# print(2+int(m/2),2+m%2)

class FC:
    def __init__(self,n,learningRate):
        self.n = n
        self.learningRate = learningRate
        self.weights = []
        self.bias = []
        self.dw = []
        self.db = []
        self.da = []
        self.a = []
        self.prev = []
        pass

    def initializeWeight(self,prevShape):
        self.prev = prevShape
        b = np.prod(prevShape)
        self.weights = np.random.randn(self.n,b)*np.sqrt(2/b)
        # self.weights = self.weights/100
        self.dw = np.zeros((self.n,b))
        self.db = np.zeros((self.n,1))
        self.bias = np.zeros((self.n,1)) # np.random.randn(self.n)
        pass

    
    def forward(self,a1):
        
        # b = np.reshape(a1,())
        d = int(np.prod(a1.shape)/a1.shape[0])
        self.a = np.reshape(a1,(a1.shape[0],d))
        
        b = self.a 
        self.prev = a1.shape

        result = np.dot(self.weights,b.T)
        # c = np.tile(self.bias,a1.shape[0])
        result = result + self.bias
        '''
        b = a1.flatten()
        d = np.prod(self.prev)
        b = np.reshape(a1,(d,a1.shape[0]))
        b = b.T
        result = np.dot(b,self.weights)
        c = np.tile(self.bias,(a1.shape[0],1))
        result = result + c'''
        
        return result.T
        pass

    
    def backward(self,result):
        totalSample = result.shape[0]
        
        # result = np.reshape(result,(result.shape[0],1))
        self.dw = (1/totalSample)*np.dot(result.T,self.a)
        self.db = (1/totalSample)*np.sum(result.T,axis=0)
        self.da = np.dot(self.weights.T,result.T)
        self.db.resize(self.bias.shape)
        self.updateWeight()

       
        # self.prev.insert(0,self.a.shape[0])
        self.da = np.reshape(self.da.T,self.prev)
        return self.da # np.reshape(self.da,self.prev)
        pass
    
    
    def updateWeight(self):
        # print(self.weights.shape,self.bias.shape)
        # print(self.dw)
        # print("#################")
        # print(self.db)
        # print("#################")
        self.weights = self.weights - self.learningRate*self.dw
        self.bias = self.bias - self.learningRate * self.db
        # norm = np.linalg.norm(self.weights)
        # self.weights = self.weights/norm
        # print(self.weights)
        # print("#################")
        # print(self.bias)
        # print("#################")
        pass

# r = np.random.randn(10,12)
# fc = FC(10,0.01)
# prev=[1,1,100]
# fc.initializeWeight(prev)
# res = fc.forward(np.random.randn(12,1,1,100))
# # print(res.shape)
# res = fc.backward(r)
# print(res.shape)

class ReLU:
    def forward(self,a1):
        a1[a1<0] = 0
        self.a = a1
        
        return a1
        pass

    def backward(self,a1):
        # result = np.zeros(a1.shape)
        # mask = self.a<0
        self.a[self.a>0] = 1
        result = np.multiply(self.a,a1)
        # print(a1.shape)
        # for i in range(a1.shape[0]):
        #     for j in range(a1.shape[1]):
        #         for k in range(a1.shape[2]):
        #             for l in range(a1.shape[3]):
        #                 if self.a[i,j,k,l] > 0:
        #                     result[i,j,k,l] = a1[i,j,k,l]

        return result
        pass

class Softmax:
    def __init__(self,n):
        self.n = n
        # self.learningRate = learningRate
        # self.weights = []
        self.da = []
        self.a = []
        pass

    def initializeWeight(self,prevShape):
        # b = np.prod(prevShape)
        # self.weights = np.random.randn(self.n,b)
        pass

    def forward(self,a1):
        r = np.exp(a1) #- np.max(a1))
        
        # result = r/np.sum(r,axis=0)
        result = np.zeros(r.shape)
        for i in range(r.shape[0]):
            s = np.sum(r[i])
            result[i] = r[i]/s

        self.a = result
        return result
        pass

    def backward(self,result):
        self.da = self.a - result
        return self.da
        pass
    def updateWeight(self):
        pass


def crossEntrophyCost(y,yPredicted):
    # print(y.shape)
    # print(yPredicted.shape)
    
    # cost = (-1/yPredicted.shape[0]) * (np.dot(y, np.log(yPredicted).T) + np.dot((1-y), np.log(1-yPredicted).T))
    # cost = np.sum(cost,axis=0)

    yHat = np.log(yPredicted)
    cost = -1*np.multiply(y,yHat)
    cost = np.sum(cost)
    cost = cost/yPredicted.shape[0]
    return cost

# c = Conv(5,1,1,0)
# a1 = np.ones((3,28,28))
# a1 = np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
# print(a1.shape)
# c.initializeWeight(a1.shape[0])
# print(c.forward(a1))

# cnn = CNN()
# a = np.zeros(4)
# df = pd.read_csv('/home/abser/Desktop/4-2(own)/CSE 472/Offline 3/ToyDataset/trainNN.txt', delimiter="\s+", header=None)
# X_train = df[df.columns[:-1]].values
# y_train = df[df.columns[-1]].values
# print(X_train.shape)


np.random.seed(8)

def defineModel(dataset=1,learnRate = 0.01):
    layers = []
    if dataset==1:
        prevShape = [28,28,1]
        totalOutputClass = 10
    if dataset==2:
        prevShape = [32,32,3]
        totalOutputClass = 10


    with open("input.txt",'r') as f:
        for line in f:
            params = line.strip().split(' ')
            if params[0]=='Conv':
                n,f,s,p = (params[1:])
                c = Conv(int(n),int(f),int(s),int(p),learnRate)
                c.initializeWeight(prevShape[2],np.prod(prevShape))
                layers.append(c)
                dim1 = int((prevShape[0] + 2*(int)(p) - (int)(f))/(int)(s) + 1)
                dim2 = int((prevShape[1] + 2*(int)(p) - (int)(f))/(int)(s) + 1)
                prevShape = [dim1,dim2,int(n)]

            elif params[0]=='ReLU':
                r = ReLU()
                layers.append(r)

            elif params[0]=='Pool':
                f,s = params[1:]
                pool = Pooling(int(f),int(s))
                dim1 = int((prevShape[0] - (int)(f))/(int)(s) + 1)
                dim2 = int((prevShape[1] - (int)(f))/(int)(s) + 1)
                prevShape = [dim1,dim2,prevShape[2]]
                layers.append(pool)

            elif params[0]=='FC':
                n = params[1]
                fc = FC(int(n),learnRate)
                fc.initializeWeight(prevShape)
                prevShape = [int(n),1]
                layers.append(fc)

            elif params[0]=='Softmax':
                soft = Softmax(totalOutputClass)
                # soft.initializeWeight(prevShape)
                layers.append(soft)
    return layers
 
def runModel(layers,dataset=1,numberOfEpoch = 5,batchSize = 100):
    if dataset==1:
        train_X,train_Y,train_Y_encoded,test_X,test_Y,test_Y_encoded,validation_X,validation_Y,validation_Y_encoded=readMnistData()
    else:
        train_X,train_Y,train_Y_encoded,test_X,test_Y,test_Y_encoded,validation_X,validation_Y,validation_Y_encoded=readCifarData()
    
    for itr in range(numberOfEpoch):
        output = []
        first = []
        last = []
        totalBatch = int(train_X.shape[0]/batchSize)
        for i in range(totalBatch):

            start = i*batchSize
            end = (i+1)*batchSize

            inp = train_X[start:end]

            for j in range(len(layers)):
                inp = layers[j].forward(inp)
                # flat = inp.flatten()
                # print("Layer ",j+1)
                # print(flat[:5])

            res = train_Y_encoded[start:end]
            
            for j in reversed(range(len(layers))):
                res = layers[j].backward(res)
                # flat = res.flatten()
                # print("Layer ",j+1)
                # print(flat[:5]) 
            
            if (i+1)%200==0:
                out = np.argmax(inp,axis=1)
                loss = crossEntrophyCost(train_Y_encoded[start:end],inp)
                acc = accuracy_score(train_Y[start:end],out)*100
                f1 = f1_score(train_Y[start:end],out,average='macro')*100
                print("Epoc {}, Batch {}  loss= {:.4f} Accuracy = {:.2f}% f1(macro) = {:.2f}%".format(itr+1,i+1,loss,acc,f1))


        inp = validation_X

        for j in range(len(layers)):
            inp = layers[j].forward(inp)

        yPredicted = np.argmax(inp,axis=1)
        loss = crossEntrophyCost(validation_Y_encoded,inp)
        accuracy = accuracy_score(validation_Y,yPredicted)*100
        f1 = f1_score(validation_Y,yPredicted,average='macro')*100

        print('Report of Validation Data')
        print('epoch ',itr+1)
        print('loss = {:.4f}, accuracy ={:.2f}%, f1_score(macro) = {:.2f}%'.format(loss,accuracy,f1))
        print('\n\n')


        target_names = [0,1,2,3,4,5,6,7,8,9]
        print(classification_report(validation_Y, yPredicted, labels=list(range(10))))

    inp = test_X


    for j in range(len(layers)):
        inp = layers[j].forward(inp)

    yPredicted = np.argmax(inp,axis=1)
    loss = crossEntrophyCost(test_Y_encoded,inp)
    accuracy = accuracy_score(test_Y,yPredicted)*100
    f1 = f1_score(test_Y,yPredicted,average='macro')*100
    print('Report of Test Data')
    print('loss = {:.4f}, accuracy ={:.2f}%, f1_score(macro) = {:.2f}%\n\n'.format(loss,accuracy,f1))

    target_names = [0,1,2,3,4,5,6,7,8,9]
    print(classification_report(test_Y, yPredicted,labels=list(range(10))))


layers = defineModel(dataset=1,learnRate=0.001)
runModel(layers,dataset=1,numberOfEpoch=2,batchSize=16)
