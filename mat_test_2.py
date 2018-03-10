import numpy as np
from scipy.optimize import linprog

A = np.array( [[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
               [-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,1,0,0,0],
               [0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,1,0,0],
               [0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,1,0],
               [0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,1],
               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1]])


c = np.random.random(20)

b = np.zeros(9)
b[[0,2]] = 1
b[8] = -2

lambda_ = np.ones(20)
lambda_[[0, 5, 16,17]] = 0

for i in range(200):
    res = linprog(c, -np.eye(20), np.zeros(20), A, b )
    f = res.x
    dual_res = linprog(- b, A.T, c)
    lam = A.T.dot(dual_res.x) + c
    np.set_printoptions(threshold=np.nan)
    
    M = np.zeros((49,49))
    M[0:20,20:40] = -np.diag(lam)
    M[0:20, 40: 50] = A.T
    M[20:40, 0:20] = - np.eye(20)
    M[20:40, 20:40] = - np.diag(f)
    M[40:50, 0:20] = A
    M[0:20, 0:20] = np.eye(20)
    
    
    BB_last = np.array([0,0,0,1])
    #print('mask: ', BB_last)
    #print('Flow last frame: ' , f[-4:])
    loss = (np.ones(4)- 2*BB_last).dot(f[-4:])
    
    dldf = np.zeros(20)
    dldf[-4:] = (np.ones(4) - 2* BB_last)
    
    e = np.zeros(49)
    e[0:20] = dldf
    #print('dldf: ', dldf)
    
    res_diff = np.linalg.lstsq(M, e)
    dldc = - res_diff[0][0:20]
    #print('dldc: ', dldc)
    c -= 10**14 * dldc # vllt transponieren
    print('Loss: ' , loss)
    print('costs', np.round(c, decimals = 2))
    print('flow', f)
    i +=1

