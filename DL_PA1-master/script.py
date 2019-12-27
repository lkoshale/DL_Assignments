import numpy as np
import math

#parameters take from arguments
lr=0.1          #learning rate
momentum=1
num_hidden=3
sizes=[2,2,2]
activation="sigmoid"
loss="ce"
opt="gd"
batch_size=400
epochs=100
anneal="true"
save_dir="/"
expt_dir="/"
train="dl2019pa1/small.csv"
val="dl2019pa1/valid.csv"
test="/"

sizes.insert(0,784)
sizes.append(10)
L = len(sizes)

Weights=[]
Baises =[]
Hidden = []  #H(0) = 1st hidden val

#m*784 x 784*1(x)
for i in range(0,L-1):
    m = sizes[i]
    n = sizes[i+1]

    if m>0 and n>0:
        w_init = 0.01*np.random.normal(0,1,(n,m))        #normal distribution # resize
        Weights.append(w_init)
        Baises.append(np.random.normal(0,0.1,n))
    else:
        print("size are negative !! terminate ")
        exit(0)
        pass


#print(Weights)
#read from file
train_data = np.genfromtxt(train, delimiter=',', skip_header=1)
train_Y = train_data[:,785]
train_X = train_data[:,1:785]
#print(train_X[0])


def sigmoid(z):
    den = 1 + np.exp(-z)
    return (1 / den)


def softmax(z):
    return ( np.exp(z)/(np.sum(np.exp(z))))

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
    e_l[y]=1
    # return -(e(l) - y' )
    return -(e_l - yHat)


def gradient_g(i,H,B,x):
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

    iterator = np.flip(range(0,L-1))

    for k in iterator:
        LW_k = mult_11d(L_theta_a_k,H[k-1])
        BW_k = L_theta_a_k
        if k > 0:
            L_theta_h_k = np.matmul(W[k].transpose(), L_theta_a_k)
           # print("next ltH {} - {}".format(W[k].shape,L_theta_h_k))
            L_theta_a_k =  np.multiply(L_theta_h_k,gradient_g(k-1,H,B,x))
          #  print("next a_h {}".format(L_theta_a_k))
        deltaLB.append(BW_k)
        deltaLW.append(LW_k)

    deltaLW = np.flip(deltaLW)
    deltaLB = np.flip(deltaLB)

    return (deltaLW , deltaLB)

x = np.array([2,3,4])
W = [np.ones((2,3)),np.ones((3,2)),np.ones((2,3)) ]
print(W)
b = [np.ones(2),np.ones(3),np.ones(2)]
y = 1
#L = 3
yHat,hidden=forward_prop(x,4,W,b,sigmoid,softmax)

print(hidden)

print(gradient_last_layer(y,yHat))
print(back_prop(W,b,4,hidden,yHat,y,x))