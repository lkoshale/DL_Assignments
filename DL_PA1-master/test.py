
import numpy as np
from sklearn.metrics import accuracy_score

lr=0.09       #learning rate
num_hidden=1
sizes=[4]
batch_size=2
epochs= 1000

sizes.insert(0,2)
sizes.append(2)
L = len(sizes)


Weights=[]
Baises =[]
Hidden = []  #H(0) = 1st hidden val

np.random.seed(1234)

#m*784 x 784*1(x)
for i in range(0,L-1):
    m = sizes[i]
    n = sizes[i+1]

    if m>0 and n>0:
        # val = 1 / np.sqrt(m)
        w_init = 0.01*np.random.normal(0,1,(n,m))        #normal distribution # resize
        Weights.append(w_init)
        Baises.append(0.01*np.random.normal(0,1,n) )
    else:
        print("size are negative !! terminate ")
        exit(0)
        pass

print(Weights)
print(Baises)

data = np.genfromtxt("xor.txt",delimiter=",")
train_X = data[:,:2]
train_Y = data[:,2]

print(train_X.shape)
print(train_Y.shape)

def sigmoid(z):
    # if -z > np.log(np.finfo(dtype=float).max):
    #     return np.zeros(z.shape)
    a = []
    for i in range(0,len(z)):
        if -z[i] > np.log(np.finfo(dtype=float).max):
            a.append(0.0)
        else:
            a.append((1/(1+np.exp(-z[i]))))

    return np.array(a)
    # den = 1 + np.exp(-z)
    # return (1 / den)


def softmax(z):
    return ( np.exp(z)/(np.sum(np.exp(z))))


def get_yHat(X, L, W, B, g, O):
    Y = []
    for x in X:
        y, H = forward_prop(x, L, W, B, g, O)
        Y.append(np.argmax(y))

    return np.array(Y)

def loss_fn(x,W,b,Y):
    Yhat, hidden = forward_prop(x, L, W, b, sigmoid, softmax)
    loss = -1* np.log(Yhat[int(Y)])
    return loss

#for w11
def approx(x,W,b,Y,k,i,j):
    W1 = W.copy()
    W1[k][i][j] = W[k][i][j] + 0.00001
    l1 = loss_fn(x,W1,b,Y)
    W1[k][i][j] = W[k][i][j] - 0.00001
    l2 = loss_fn(x,W1,b,Y)
    return ( l1 - l2)/0.00002




#forward prop for one data point
# x input layer value
def forward_prop(x,L,W,b,g,O):
    hidden=[]
    h_k = x
    for k in range(0,L-2):
       # print("shape : {} , {}".format(W[k].shape,h_k.shape))
        a_k = b[k] + W[k].dot(h_k)
        h_k = g(a_k)
        hidden.append(h_k)

    a_L = b[L-2] + W[L-2].dot(h_k)
    return ( O(a_L),hidden )



#for somftmax fun
def gradient_last_layer(y,yHat):
    e_l = np.zeros(len(yHat))
    e_l[int(y)]=1
    # return -(e(l) - y' )
    return -(e_l - yHat)


def gradient_g(i,W,H,B,x):
    if(i>0):
        a_i = B[i] + W[i].dot(H[i-1])
    else:
        a_i = B[i] + W[i].dot(x)
    return sigmoid(a_i)*( 1 - sigmoid(a_i))

def mult_11d(first,second):
    array=np.array([])
    for i in range(0,len(first)):
        if i > 0:
            array = np.vstack((array,np.multiply(second,first[i])))
        elif i == 0 :
            array = np.multiply(second,first[i])
    return array



#this function return deltaLW and deltaLb
# Y_hat is the output of forward prop
def back_prop(W,B,L,H,Y_hat,Y,x):
    deltaLW = []
    deltaLB = []
    #output gradient
    L_theta_a_k = gradient_last_layer(Y,Y_hat)
    # print(L_theta_a_k.shape)
    # input()

    iterator = np.flip(range(0,L-1))

    for k in iterator:
        if k > 0:
            LW_k = np.outer(L_theta_a_k,H[k-1])   # LW_k = mult_11d(L_theta_a_k,H[k-1])
        else:
            LW_k = np.outer(L_theta_a_k,x)       #LW_k = mult_11d(L_theta_a_k,x)

        # print(LW_k.shape)
        # print(H[k-1].shape)
        # print(LW_k,L_theta_a_k,H[k-1])
        # input()
        BW_k = L_theta_a_k
        if k > 0:
            L_theta_h_k = np.matmul(W[k].transpose(), L_theta_a_k)
            # print(W[k].transpose().shape)
            # print(L_theta_h_k.shape,L_theta_h_k)

            #input()
            # print("next ltH {} - {}".format(W[k].shape,L_theta_h_k))
            L_theta_a_k =  np.multiply(L_theta_h_k,gradient_g(k-1,W,H,B,x))
           # print(L_theta_a_k.shape,gradient_g(k-1,W,H,B,x).shape)

            # print("next g_g : {}".format(gradient_g(k-1,W,H,B,x)))
        # print(L_theta_a_k)
        deltaLB.append(BW_k)
        deltaLW.append(LW_k)





    deltaLW = np.flip(deltaLW)
    deltaLB = np.flip(deltaLB)

    # print(" delta W : {}".format(deltaLW))
    # exit(0)

    return (deltaLW , deltaLB)

def grad_descent_mini_batch(W,b):
    for i in range(epochs):
        num_points = 0
        dW = [Wx*0 for Wx in W]
        dB = [bx*0 for bx in b]
        # print(W)
        for x,y in zip(train_X,train_Y):
            Yhat,hidden = forward_prop(x,L,W,b,sigmoid,softmax)
            # print(Yhat)
            # print(hidden)
            (dw,db) = back_prop(W,b,L,hidden,Yhat,y,x)
            # print(dw)
            # print("approx w000:")
            # print(approx(x,W,b,y,1,0,1))

            dW= [ dWx + dwx for dWx,dwx in zip(dW,dw) ]
            dB= [ dBx + dbx for dBx,dbx in zip(dB,db) ]
            num_points += 1


            if num_points % batch_size == 0 or num_points==train_X.shape[0] :
                dW = [ dWx/batch_size for dWx in dW ]
                dB = [ dBx/batch_size for dBx in dB ]
                W =  [ Wx - lr*dWx for Wx,dWx in zip(W,dW) ]
                b =  [ bx - lr*dBx for bx,dBx in zip(b,dB) ]
                # print(W)
                dW = [Wx * 0 for Wx in W]
                dB = [bx * 0 for bx in b]


        print("accuracy : {}".format(accuracy_score(y_pred=get_yHat(train_X, L, W, b, sigmoid, softmax), y_true=train_Y)))

    return (W,b)





(W,B)= grad_descent_mini_batch(Weights,Baises)
Y_Pred = get_yHat(train_X,L,W,B,sigmoid,softmax)

s=0
for i in range(0,len(Y_Pred)):
   if train_Y[i] != Y_Pred[i]:
       s+=1

print(s)
print(Y_Pred)
print(train_Y)


print("accuracy : {}".format(accuracy_score(y_pred=Y_Pred,y_true=train_Y)))