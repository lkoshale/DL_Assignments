import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import argparse
import math
import pickle as pkl

# parameters take from arguments
lr = 0.3  # learning rate
momentum = 0.9
num_hidden = 2
sizes = [100, 100]
activation = "sigmoid"
loss = "ce"
opt = "gd"
batch_size = 20
epochs = 5
anneal = "true"
save_dir = "/"
expt_dir = "/"
train = "dl2019pa1/train.csv"
val = "dl2019pa1/valid.csv"
test = "dl2019pa1/test.csv"
save = False
plot = True

train_loss_epoch =[]
val_loss_epoch=[]


Weights = []
Baises = []
Hidden = []  # H(0) = 1st hidden val

np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float)
parser.add_argument("--num_hidden", type=int)
parser.add_argument("--sizes", type=str)
parser.add_argument("--activation", type=str, choices=["tanh", "sigmoid"])
parser.add_argument("--loss", type=str, choices=["ce", "sq"])
parser.add_argument("--opt", type=str, choices=["gd", "momentum", "nag", "adam"])
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--anneal", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--expt_dir", type=str)
parser.add_argument("--train", type=str)
parser.add_argument("--val", type=str)
parser.add_argument("--test", type=str)

args = parser.parse_args()

if args.lr is not None:
    lr = args.lr

if args.momentum is not None:
    momentum = args.momentum

if args.num_hidden is not None:
    num_hidden = args.num_hidden

if args.sizes is not None:
    sizes = args.sizes.split(',')
    sizes = [int(x) for x in sizes]

if args.activation is not None:
    activation = args.activation

if args.loss is not None:
    loss = args.loss

if args.opt is not None:
    opt = args.opt

if args.batch_size is not None:
    batch_size = args.batch_size

if args.epochs is not None:
    epochs = args.epochs

if args.anneal is not None:
    anneal = args.anneal

if args.save_dir is not None:
    save_dir = args.save_dir

if args.expt_dir is not None:
    expt_dir = args.expt_dir

if args.train is not None:
    train = args.train

if args.val is not None:
    val = args.val

if args.test is not None:
    test = args.test

#######################################

sizes.insert(0, 100)
sizes.append(10)
L = len(sizes)

# m*784 x 784*1(x)
for i in range(0, L - 1):
    m = sizes[i]
    n = sizes[i + 1]

    if m > 0 and n > 0:
        # val = 1 / np.sqrt(m)
        w_init = 0.01 * np.random.normal(0, 1, (n, m))  # normal distribution # resize
        Weights.append(w_init)
        Baises.append(0.01 * np.random.normal(0, 1, n))
    else:
        print("size are negative !! terminate ")
        exit(0)
        pass

# print(Weights)
# read from file
# train_data = np.genfromtxt(train, delimiter=',', skip_header=1)
# train_Y = train_data[:,785]
# train_X = train_data[:,1:785]
# #print(train_X[0])

DF = pd.read_csv(train)
# DF = DF.sample(frac=1).reset_index(drop=True) #shuffle the data
train_X = DF.iloc[:1000, 1:785].to_numpy()

train_Y = DF.iloc[:1000, 785].to_numpy()

DF_val = pd.read_csv(val)
val_X = DF_val.iloc[:100, 1:785].to_numpy()
val_Y = DF_val.iloc[:100, 785].to_numpy()

DF_test = pd.read_csv(test)
test_X = DF_test.iloc[:100, 1:].to_numpy()



pca = PCA(n_components=100, random_state=1234)
pca.fit(train_X)
train_X = pca.transform(train_X)
val_X = pca.transform(val_X)
test_X = pca.transform(test_X)

print(val_X.shape)
print(val_Y.shape)


def sigmoid(z):
    a = []
    for i in range(0, len(z)):
        if -z[i] > np.log(np.finfo(dtype=float).max):
            a.append(0.0)
        else:
            a.append((1 / (1 + np.exp(-z[i]))))

    return np.array(a)
    # den = 1 + np.exp(-z)
    # return (1 / den)


def softmax(z):
    return (np.exp(z) / (np.sum(np.exp(z))))


def tanh(z):
    return ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)))


def get_yHat(X, L, W, B, g, O):
    Y = []
    for x in X:
        y, H, A = forward_prop(x, L, W, B, g, O)
        Y.append(np.argmax(y))

    return np.array(Y)


# forward prop for one data point
# x input layer valu
def forward_prop(x, L, W, b, g, O):
    hidden = []
    activated = []
    h_k = x
    for k in range(0, L - 2):
        # print("shape : {} , {}".format(W[k].shape,h_k.shape))
        a_k = b[k] + W[k].dot(h_k)
        h_k = g(a_k)
        hidden.append(h_k)
        activated.append(a_k)

    a_L = b[L - 2] + W[L - 2].dot(h_k)

    return (O(a_L), hidden, activated)


def get_Yhat_loss(X, L, W, B, g, O, Y):
    loss_sum = 0
    Yhat = []
    for i in range(0, X.shape[0]):
        y, H, A = forward_prop(X[i], L, W, B, g, O)
        loss_sum += -1 * np.log(y[Y[i]])
        Yhat.append(np.argmax(y))

    return (loss_sum / X.shape[0], np.array(Yhat))


# for somftmax fun
def gradient_last_layer(y, yHat):
    e_l = np.zeros(len(yHat))
    e_l[y] = 1
    # return -(e(l) - y' )
    return -(e_l - yHat)


def gradient_g(i, A):
    return sigmoid(A[i]) * (1 - sigmoid(A[i]))


def gradient_tanh(i, A):
    return (1 - np.square(tanh(A[i])))


def mult_11d(first, second):
    array = np.array([])
    for i in range(0, len(first)):
        if i > 0:
            array = np.vstack((array, np.multiply(second, first[i])))
        elif i == 0:
            array = np.multiply(second, first[i])
    return array


# this function return deltaLW and deltaLb
# Y_hat is the output of forward prop
def back_prop(W, B, L, H, Y_hat, Y, A, x):
    deltaLW = []
    deltaLB = []
    # output gradient
    L_theta_a_k = gradient_last_layer(Y, Y_hat)

    iterator = np.flip(range(0, L - 1))

    for k in iterator:
        if k > 0:
            LW_k = mult_11d(L_theta_a_k, H[k - 1])
        else:
            LW_k = mult_11d(L_theta_a_k, x)
        BW_k = L_theta_a_k
        if k > 0:
            L_theta_h_k = np.matmul(W[k].transpose(), L_theta_a_k)
            # print("next ltH {} - {}".format(W[k].shape,L_theta_h_k))
            L_theta_a_k = np.multiply(L_theta_h_k, gradient_g(k - 1, A))
        #  print("next a_h {}".format(L_theta_a_k))
        deltaLB.append(BW_k)
        deltaLW.append(LW_k)

    deltaLW = np.flip(deltaLW)
    deltaLB = np.flip(deltaLB)

    # print(deltaLW[0].shape)

    return (deltaLW, deltaLB)


def grad_descent_mini_batch(W, b):
    for i in range(epochs):
        num_points = 0
        dW = [Wx * 0 for Wx in W]
        dB = [bx * 0 for bx in b]
        steps = 0
        for x, y in zip(train_X, train_Y):
            Yhat, hidden, A = forward_prop(x, L, W, b, sigmoid, softmax)
            (dw, db) = back_prop(W, b, L, hidden, Yhat, y, A, x)

            dW = [dWx + dwx for dWx, dwx in zip(dW, dw)]
            dB = [dBx + dbx for dBx, dbx in zip(dB, db)]
            num_points += 1

            if num_points % batch_size == 0 or num_points == train_X.shape[0]:
                W = [Wx - lr * (dWx / batch_size) for Wx, dWx in zip(W, dW)]
                b = [bx - lr * (dBx / batch_size) for bx, dBx in zip(b, dB)]
                # print(dW)
                dW = [Wx * 0 for Wx in W]
                dB = [bx * 0 for bx in b]
                steps += 1
                # print(steps)
                if steps % 20 == 0:
                    loss, y_pred = get_Yhat_loss(train_X, L, W, b, sigmoid, softmax, train_Y)
                    accuracy = accuracy_score(y_pred, train_Y)
                    print("Epoch {}, Step {}, Loss: {}, Error:{}, lr: {}".format(i, steps, format(loss, '.2f'),
                                                                                 format((1 - accuracy) * 100, '0.2f'),
                                                                                 lr))
                    if plot and num_points == train_X.shape[0]:
                        loss2,y_pred2 = get_Yhat_loss(val_X, L, W, b, sigmoid, softmax, val_Y)
                        train_loss_epoch.append(loss)
                        val_loss_epoch.append(loss2)



    return (W, b)


def momentum_grad_descent_mini_batch(W, b, gamma):
    prev_w = [Wx * 0 for Wx in W]
    prev_b = [bx * 0 for bx in b]
    for i in range(epochs):
        num_points = 0
        steps = 0
        dW = [Wx * 0 for Wx in W]
        dB = [bx * 0 for bx in b]
        for x, y in zip(train_X, train_Y):
            Yhat, hidden, A = forward_prop(x, L, W, b, sigmoid, softmax)
            (dw, db) = back_prop(W, b, L, hidden, Yhat, y, A, x)
            dW = [dWx + dwx for dWx, dwx in zip(dW, dw)]
            dB = [dBx + dbx for dBx, dbx in zip(dB, db)]
            num_points += 1

            if num_points % batch_size == 0 or num_points == train_X.shape[0]:
                v_w = [gamma * prev_wx + lr * (dWx / batch_size) for prev_wx, dWx in zip(prev_w, dW)]
                v_b = [gamma * prev_bx + lr * (dBx / batch_size) for prev_bx, dBx in zip(prev_b, dB)]

                W = [Wx - v_wx for Wx, v_wx in zip(W, v_w)]
                b = [bx - v_bx for bx, v_bx in zip(b, v_b)]

                prev_w = v_w
                prev_b = v_b

                dW = [Wx * 0 for Wx in W]
                dB = [bx * 0 for bx in b]
                steps+=1

                if steps % 20 == 0:
                    loss, y_pred = get_Yhat_loss(train_X, L, W, b, sigmoid, softmax, train_Y)
                    accuracy = accuracy_score(y_pred, train_Y)
                    print("Epoch {}, Step {}, Loss: {}, Error:{}, lr: {}".format(i, steps, format(loss, '.2f'),
                                                                                 format((1 - accuracy) * 100, '0.2f'),
                                                                                 lr))
                if plot and num_points == train_X.shape[0]:
                    loss, y_pred = get_Yhat_loss(train_X, L, W, b, sigmoid, softmax, train_Y)
                    loss2, y_pred2 = get_Yhat_loss(val_X, L, W, b, sigmoid, softmax, val_Y)
                    train_loss_epoch.append(loss)
                    val_loss_epoch.append(loss2)
                    

    return (W, b)


def nestrov_grad_descent_mini_batch(W, b, gamma):
    prev_w = [Wx * 0 for Wx in W]
    prev_b = [bx * 0 for bx in b]
    for i in range(epochs):
        num_points = 0
        dW = [Wx * 0 for Wx in W]
        dB = [bx * 0 for bx in b]
        for x, y in zip(train_X, train_Y):
            W_pass = [Wx - gamma * prev_wx for Wx, prev_wx in zip(W, prev_w)]
            b_pass = [bx - gamma * prev_bx for bx, prev_bx in zip(b, prev_b)]
            Yhat, hidden, A = forward_prop(x, L, W_pass, b_pass, sigmoid, softmax)
            (dw, db) = back_prop(W_pass, b_pass, L, hidden, Yhat, y, A, x)
            dW = [dWx + dwx for dWx, dwx in zip(dW, dw)]
            dB = [dBx + dbx for dBx, dbx in zip(dB, db)]
            num_points += 1

            if num_points % batch_size == 0:
                v_w = [gamma * prev_wx + lr * (dWx / batch_size) for prev_wx, dWx in zip(prev_w, dW)]
                v_b = [gamma * prev_bx + lr * (dBx / batch_size) for prev_bx, dBx in zip(prev_b, dB)]

                W = [Wx - v_wx for Wx, v_wx in zip(W, v_w)]
                b = [bx - v_bx for bx, v_bx in zip(b, v_b)]
                prev_w = v_w
                prev_b = v_b
                dW = [Wx * 0 for Wx in W]
                dB = [bx * 0 for bx in b]

    return (W, b)


def adam_mini_batch(W,b,beta1,beta2,eps):
    m_w = [Wx * 0 for Wx in W]
    m_b = [bx * 0 for bx in b]
    v_w = [Wx * 0 for Wx in W]
    v_b = [bx * 0 for bx in b]

    for i in range(epochs):
        num_points = 0
        dW = [Wx * 0 for Wx in W]
        dB = [bx * 0 for bx in b]
        steps = 0
        for x, y in zip(train_X, train_Y):
            Yhat, hidden,A = forward_prop(x, L, W, b, sigmoid, softmax)
            (dw, db) = back_prop(W, b, L, hidden, Yhat, y,A,x)
            dW = [dWx + dwx for dWx, dwx in zip(dW, dw)]
            dB = [dBx + dbx for dBx, dbx in zip(dB, db)]
            num_points += 1

            if num_points % batch_size == 0 or num_points == train_X.shape[0]:

                dW = [dWx/batch_size for dWx in dW]
                dB = [dBx/batch_size for dBx in dB]

                m_w = [m_wx * beta1 + (1 - beta1) * dWx for m_wx, dWx in zip(m_w, dW)]
                m_b = [m_bx * beta1 + (1 - beta1) * dBx for m_bx, dBx in zip(m_b, dB)]

                v_w = [v_wx*beta2 + (1-beta2) * np.linalg.norm(dWx)** 2 for v_wx,dWx in zip(v_w,dW)]
                v_b = [v_bx * beta2 + (1 - beta2) * np.linalg.norm(dBx) ** 2 for v_bx, dBx in zip(v_b, dB)]

                m_what = [m_wx/(1 - math.pow(beta1,i+1)) for m_wx in m_w]
                m_bhat = [m_bx / (1 - math.pow(beta1, i + 1)) for m_bx in m_b]

                v_what = [v_wx / (1 - math.pow(beta2, i + 1)) for v_wx in v_w]
                v_bhat = [v_bx / (1 - math.pow(beta2, i + 1)) for v_bx in v_b]

                W = [ Wx - lr * m_whatx / (np.sqrt( v_whatx + eps )) for Wx, m_whatx, v_whatx in zip( W, m_what, v_what) ]
                b = [ Bx - lr * m_bhatx / (np.sqrt( v_bhatx + eps )) for Bx, m_bhatx, v_bhatx in zip(b, m_bhat, v_bhat) ]

                dW = [Wx * 0 for Wx in W]
                dB = [bx * 0 for bx in b]

                steps+=1

                if steps % 20 == 0:
                    loss, y_pred = get_Yhat_loss(train_X, L, W, b, sigmoid, softmax, train_Y)
                    accuracy = accuracy_score(y_pred, train_Y)
                    print("Epoch {}, Step {}, Loss: {}, Error:{}, lr: {}".format(i, steps, format(loss, '.2f'),
                                                                                 format((1 - accuracy) * 100, '0.2f'),
                                                                                 lr))
                if plot and num_points == train_X.shape[0]:
                    loss, y_pred = get_Yhat_loss(train_X, L, W, b, sigmoid, softmax, train_Y)
                    loss2,y_pred2 = get_Yhat_loss(val_X, L, W, b, sigmoid, softmax, val_Y)
                    train_loss_epoch.append(loss)
                    val_loss_epoch.append(loss2)
                


    return (W,b)


# (W, B) = grad_descent_mini_batch(Weights, Baises)
# (W,B)= momentum_grad_descent_mini_batch(Weights,Baises,momentum)
(W,B)= adam_mini_batch(Weights,Baises,0.9,0.999,0.00001)
Y_Pred = get_yHat(train_X, L, W, B, sigmoid, softmax)
Y_val_pred = get_yHat(val_X, L, W, B, sigmoid, softmax)
Y_test_pred = get_yHat(test_X, L, W, B, sigmoid, softmax)

with open("submission.csv", 'w') as f:
    f.write("id,label\n")
    for i in range(0, Y_test_pred.shape[0]):
        f.write("{},{}\n".format(i,Y_test_pred[i]))

s = 0
for i in range(0, len(Y_Pred)):
    if train_Y[i] != Y_Pred[i]:
        s += 1

print(s)
print(Y_Pred)
print(train_Y)

if save == True:
    with open('W.pkl', 'wb') as f:
        pkl.dump(W, f)

    with open('b.pkl', 'wb') as f:
        pkl.dump(B, f)


if plot==True:
    with open("train_loss_plot.txt",'w') as f:
        for i in range(0,len(train_loss_epoch)):
            f.write("{} {}\n".format(i+1,train_loss_epoch[i]))

    with open("val_loss_plot", 'w') as f:
        for i in range(0, len(val_loss_epoch)):
            f.write("{} {}\n".format(i + 1, val_loss_epoch[i]))


print("accuracy training : {}".format(accuracy_score(y_pred=Y_Pred, y_true=train_Y)))

print("accuracy validation : {}".format(accuracy_score(y_pred=Y_val_pred, y_true=val_Y)))
