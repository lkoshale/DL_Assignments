import numpy as np

def sigmoid(w,b,x):
	return 1/(1 + np.exp(-(w*x + b)))

def error_sig(w,b):
	error = 0
	for x,y in est : #estimated-real values
		fx = f(w,b,x)
		error += 0.5 * ( fx - y )**2
	return error

def grad_w_sig(w,b,x,y): #derivative of error corresponding to single estimate wrt w
	fx = f(w,b,x)
	return (fx - y)*fx*(1 - fx)

def grad_b_sig(w,b,x,y): #derivative of error corresponding to single estimate wrt b
	fx = f(w,b,x)
	return (fx - y)*fx*(1 - fx)*x



def grad_descent(W,b):
	for i in range(epochs):
		for x,y in zip(X,Y):
			Yhat,hidden = forward_prop(x,L,W,b,sigmoid,softmax)
			(dw,db) = back_prop(W,b,L,hidden,Yhat,y,x)
			W = W - lr*dw
			b = b - lr*db

	return (W,b)


def momentum_grad_descent(W,b):
	prev_w,prev_b = 0,0
	for i in range(epochs):
		for x,y in zip(X,Y):
			Yhat,hidden = forward_prop(x,L,W,b,sigmoid,softmax)
			(dw,db) = back_prop(W,b,L,hidden,Yhat,y,x)

			v_w = gamma*prev_w + lr*dw
			v_b = gamma*prev_b + lr*db

			w = w - v_w
			b = b - v_b

			prev_w = v_w
			prev_b = v_b

	return (W,b)

def nestrov_grad_descent(W,b):
	prev_w,prev_b = 0,0
	for i in range(epochs):		
		for x,y in zip(X,Y):
			v_w = gamma*prev_w
			v_b = gamma*prev_b
			Yhat,hidden = forward_prop(x,L,W-v_w,b-v_b,sigmoid,softmax)
			(dw,db) = back_prop(W-v_w,b-v_b,L,hidden,Yhat,y,x)

			v_w = gamma*prev_w + lr*dw
			v_b = gamma*prev_b + lr*db

			W = W - v_w
			b = b - v_b
			prev_w = v_w
			prev_b = v_b

	return (W,b)


def grad_descent_mini_batch(W,b):
	for i in range(epochs):
		num_points = 0
		for x,y in zip(X,Y):			
			Yhat,hidden = forward_prop(x,L,W,b,sigmoid,softmax)
			(dw,db) += back_prop(W,b,L,hidden,Yhat,y,x)
			num_points += 1

			if num_points % mini_batch_size == 0:
				W = W - lr*dw
				b = b - lr*db
				dw,db = 0,0

	return (W,b)

def momentum_grad_descent_mini_batch(W,b):
	prev_w,prev_b = 0,0
	for i in range(epochs):
		num_points = 0
		for x,y in zip(X,Y):
			Yhat,hidden = forward_prop(x,L,W,b,sigmoid,softmax)
			(dw,db) += back_prop(W,b,L,hidden,Yhat,y,x)
			num_points += 1

			if num_points % mini_batch_size == 0:
				v_w = gamma*prev_w + lr*dw
				v_b = gamma*prev_b + lr*db

				w = w - v_w
				b = b - v_b

				prev_w = v_w
				prev_b = v_b

				dw,db = 0,0

	return (W,b)

def nestrov_grad_descent_mini_batch(W,b):
	prev_w,prev_b = 0,0
	for i in range(epochs):
		num_points = 0
		for x,y in zip(X,Y):
			v_w = gamma*prev_w
			v_b = gamma*prev_b
			Yhat,hidden = forward_prop(x,L,W-v_w,b-v_b,sigmoid,softmax)
			(dw,db) += back_prop(W-v_w,b-v_b,L,hidden,Yhat,y,x)
			num_points += 1

			if num_points % mini_batch_size == 0:
				v_w = gamma*prev_w + lr*dw
				v_b = gamma*prev_b + lr*db

				W = W - v_w
				b = b - v_b
				prev_w = v_w
				prev_b = v_b
				dw,db = 0,0

	return (W,b)

def adagrad(W,b):
	v_w,v_b = 0,0
	for i in range(epochs):
		dw,db = 0
		for x,y in zip(X,Y):
			Yhat,hidden = forward_prop(x,L,W,b,sigmoid,softmax)
			(dw,db) += back_prop(W,b,L,hidden,Yhat,y,x)

			v_w = v_w + dw**2
			v_b = v_b + db**2

			W = W - (lr / np.sqrt(v_w + epsilon))*dw
			b = b - (lr / np.sqrt(v_b + epsilon))*db

	return (W,b)

def Adam(W,b):
	