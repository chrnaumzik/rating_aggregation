# -*- coding: utf-8 -*-
import numpy as np
def TensorMapping(T,A1,A2,A3):
    n1,n2,n3 = T.shape
    m1 = A1.shape[1]
    m2 = A2.shape[1]
    m3 = A3.shape[1]
    
    M = np.zeros((m1,m2,m3))
    
    for i1 in range(m1):
        for i2 in range(m2):
            for i3 in range(m3):
                for j1 in range(n1):
                    for j2 in range(n2):
                        for j3 in range(n3):
                            M[i1,i2,i3] = M[i1,i2,i3] + T[j1,j2,j3]*A1[j1,i1]*A2[j2,i2]*A3[j3,i3]
                            
    return M

def RobustPowerMethod(M):
    L = 100
    _iter = 100
    lambda_max = -1
    d = M.shape[0]
    vec_max = np.zeros((d,1))
    
    for i in range(L):
        ranvec = np.random.uniform(size=d).reshape(d,1) - 0.5* np.ones((d,1))
        ranvec = ranvec/np.linalg.norm(ranvec)
        for j in range(_iter):
            ranvec = TensorMapping(M,np.eye(d),ranvec,ranvec)
            ranvec = ranvec/np.linalg.norm(ranvec)
            
        _lambda = TensorMapping(M,ranvec,ranvec,ranvec).reshape(1)
        if _lambda > lambda_max:
            lambda_max = _lambda
            vec_max  = ranvec
    
    return vec_max
def AggregateCFG(mu,mode=0):
    #mode must me in [0,1,2]
    #mode == 0 : General model
    #mode == 1 : One-coin model
    #mode == 2 : Category-level one-coin model
    
    k = mu.shape[0]
    
    if mode == 0:
        newmu = mu
    elif mode == 1:
        p = np.sum(np.diag(mu))/np.sum(mu)
        newmu = np.eye(N=k)*(p-(1-p)/(k-1)) + np.ones(shape=(k,k))*(1-p)/(k-1)
    else:
        p = np.transpose(np.diag(mu))/np.sum(mu,axis=0)
        newmu = np.diag(p-(1-p)/(k-1)) + np.tile((1-p)/(k-1), (k, 1))

    return newmu

def SolveCFG(x1,x2,x3):
    k,n = x1.shape
    x1_tilde = np.matmul(np.linalg.solve(np.matmul(x2,x1.transpose())/n,np.matmul(x2,x3.transpose())/n).transpose(),x1)
    x2_tilde = np.matmul(np.linalg.solve(np.matmul(x1,x2.transpose())/n,np.matmul(x1,x3.transpose())/n).transpose(),x2)
    M2 = np.matmul(x1_tilde,x2_tilde.transpose())/n
    M3 = np.zeros((k,k,k))
    
    for i1 in range(k):
        for i2 in range(k):
            for i3 in range(k):
                M3[i1,i2,i3] = np.sum(x1_tilde[i1,:]*x2_tilde[i2,:]*x3[i3,:])/n
    
    U,S,V = np.linalg.svd(M2)
    W = np.matmul(U,np.diag(np.power(S,-0.5)))
    M3W = TensorMapping(M3,W,W,W)
    
    mu3_p = np.zeros((k,k))
    w_p = np.zeros((k,1))
    
    for c in range(k):
        vec = RobustPowerMethod(M3W).reshape((k,1))
        _lambda = TensorMapping(M3W,vec,vec,vec).reshape(1)
        for i1 in range(k):
            for i2 in range(k):
                for i3 in range(k):
                    M3W[i1,i2,i3] = M3W[i1,i2,i3] - _lambda * vec[i1]*vec[i2]*vec[i3]
        
        mu_tmp = np.matmul(_lambda*np.linalg.pinv(W.transpose()),vec)
        index = np.argmax(mu_tmp)
        if w_p[index] != 0:
            empty_slots = np.argwhere(w_p==0)[:,0]
            index = np.int(empty_slots[0].reshape(-1))
        mu3_p[:,index] = mu_tmp.reshape(-1)
        w_p[index] = np.power(1/_lambda,2)
    
    muW = np.append(mu3_p,w_p,axis=1)
    
    return muW


def spectral(A,B,N,mode=0,N_iter=50,algo = "KOS",init='MV'):      
    n = np.max(A.iloc[:,0])#number of items, i.e., restaurants
    m = np.max(A.iloc[:,1])#number of workers, i.e., users
    k = np.max(A.iloc[:,2])#number of classes, i.e., stars
    print("Number items")
    print(n)
    print("Number users")
    print(m)
    print("Number stars")
    print(k)
    if algo == "KOS":
        t = np.zeros((n,k-1))
        for l in range(k-1):
            U = np.zeros((n,m))
            for i in range(A.shape[0]):
                U[A.iloc[i,0]-1,A.iloc[i,1]-1] = 2*(A.iloc[i,2]>l+1)-1
        
            W = U - np.ones((n,1)) @ (np.ones((1,n)) @ U)/n
            U,S,V = np.linalg.svd(W)
            u = U[:,0]
            v = V[:,0]
            u = u/np.linalg.norm(u)
            v = v/np.linalg.norm(v)
            pos_index = np.where(v>=0)
            if np.sum(np.power(v[pos_index],2)) >= 0.5:
                t[:,l] = np.sign(u)
            else:
                t[:,l] = -np.sign(u)
                
        J = np.ones(n)*k

        for j in range(n):
            for l in range(k-1):
                if t[j,l] == 1:
                    J[j] = l + 1
        
        return J
    elif algo == "Ghosh":
        t = np.zeros((n,k-1))
        for l in range(k-1):
            W = np.zeros((n,m))
            for i in range(A.shape[0]):
                W[A.iloc[i,0]-1,A.iloc[i,1]-1] = 2*(A.iloc[i,2]>l+1)-1
        
            U,S,V = np.linalg.svd(W)
            u = np.sign(U[:,0])
            if np.dot(u,np.sum(W,1)) >= 0:
                t[:,l] = np.sign(u)
            else:
                t[:,l] = -np.sign(u)

        J = np.ones(n)*k

        for j in range(n):
            for l in range(k-1):
                if t[j,l] == -1:
                    J[j] = l + 1                
                    
        return J

    else:
        Z = np.zeros((n,k,m));
        for i in range(A.shape[0]):
            Z[A.iloc[i,0]-1,A.iloc[i,2]-1,A.iloc[i,1]-1] += 1
    
        group = np.mod(np.arange(0,m),3)
        Zg = np.zeros((n,k,3))

        for i in range(3):
            I = np.asarray(np.where(group == i)).reshape(-1)
            Zg[:,:,i] = np.sum(Z[:,:,I],axis=2)

        x1 = Zg[:,:,0].transpose()
        x2 = Zg[:,:,1].transpose()
        x3 = Zg[:,:,2].transpose()
        if init == "Spectral":
            muWg = np.zeros((k,k+1,3))

            muWg[:,:,0] = SolveCFG(x2,x3,x1)
            muWg[:,:,1] = SolveCFG(x3,x1,x2)
            muWg[:,:,2] = SolveCFG(x1,x2,x3)

            mu = np.zeros((k,k,m))

            for i in range(m):
                x = Z[:,:,i].transpose()
                x_alt = np.sum(Zg,axis=2).transpose() - Zg[:,:,group[i]].transpose()
                muW_alt = np.sum(muWg,axis=2) - muWg[:,:,group[i]]
                mu[:,:,i] = np.linalg.solve(np.matmul(muW_alt[:,range(k)],np.diag(muW_alt[:,k])/2),np.matmul(x_alt,x.transpose())/n).transpose()
                mu[:,:,i] = np.maximum(mu[:,:,i],1e-6)
                mu[:,:,i] = AggregateCFG(mu[:,:,i],mode)    
                for j in range(k):
                    mu[:,j,i] = mu[:,j,i] / np.sum(mu[:,j,i])
        else:
            q = np.mean(Z,2)
            q = np.divide(q,np.tile(np.sum(q,1),(1,k)).reshape(k,n).transpose())
            mu = np.zeros((k,k,m))
            for i in range(m):
                mu[:,:,i] = np.matmul(Z[:,:,i].transpose(),q)
                mu[:,:,i] = AggregateCFG(mu[:,:,i],mode)
                for c in range(k):
                    mu[:,c,i] = mu[:,c,i]/np.sum(mu[:,c,i])
                    
        TEST_NUM = 10
        for _iter in range(N_iter): 
            print(_iter)
            spectral_error = []
            idx = 0 
            q = np.zeros((n,k)) 
            for j in range(n):
                for c in range(k):
                    for i in range(m):
                        if np.dot(Z[j,:,i],mu[:,c,i])>0:
                            q[j,c] = q[j,c] + np.log(np.dot(Z[j,:,i],mu[:,c,i]))
                q[j,:] = q[j,:] - np.max(q[j,:])   
                q[j,:] = np.exp(q[j,:])
                q[j,:] = q[j,:]/np.sum(q[j,:])
                
            for i in range(q.shape[0]):
                Y_test = B.stars[(i*TEST_NUM):((i+1)*TEST_NUM)]
                y_hat = np.matmul(q[i,:],np.arange(1,k+1))
                spectral_error.append(abs(Y_test.mean()-y_hat).reshape(-1))    
                idx = idx + N[i]
            print(np.mean(spectral_error))
            for i in range(m):
                mu[:,:,i] = np.matmul(Z[:,:,i].transpose(),q)
                mu[:,:,i] = AggregateCFG(mu[:,:,i],mode)
                for c in range(k):
                    mu[:,c,i] = mu[:,c,i]/np.sum(mu[:,c,i])
    
        J = np.argmax(q,axis=1)+1   
        
        
            

    return J,q
