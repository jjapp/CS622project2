import numpy as np


def gradient_descent(grad_f, x_init, mu):
    max_grad = np.amax(grad_f(x_init))
    while max_grad > 0.0001:
        # get new grad
        new_grad = grad_f(x_init)
        for i in range(len(x_init)):
            x_init[i]=x_init[i]-mu*new_grad[i]
        max_grad = np.max(grad_f(x_init))
    return x_init


def df_test1(x):
  return np.array([2*x[0]])

if __name__=="__main__":
    x = gradient_descent(df_test1,np.array([5.0]),0.1)
    print (x)
