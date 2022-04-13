  
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import scipy.io 

def calculate_triangles(A):
    N=len(A)
    total=0
    T=[0]*5
    for i in range(N):
        for j in range(i+1,N):
            if(A[i][j]!=0):
                for k in range(j+1,N):
                    if(A[i][k]!=0 and A[j][k]!=0):
                        T[4]+=1
                        if(A[i][k]==1 and A[j][k]==1  and  A[i][j]==1):
                            T[3]+=1
                        elif((A[i][k]==1 and A[j][k]==1 and A[i][j]==-1) or (A[i][k]==1 and A[j][k]==-1 and A[i][j]==1) or (A[i][k]==-1 and A[j][k]==1 and A[i][j]==1)):
                            T[2]+=1
                        elif((A[i][k]==1 and A[j][k]==-1 and A[i][j]==-1) or (A[i][k]==-1 and A[j][k]==1 and A[i][j]==-1) or (A[i][k]==-1 and A[j][k]==-1 and A[i][j]==1)):
                            T[1]+=1
                        elif((A[i][k]==-1 and A[j][k]==-1 and A[i][j]==-1)):
                            T[0]+=1
    print(T)
    return T

def main():
    with open("./Data/wiki-election.npy", 'rb') as file:
        A = np.load(file,allow_pickle=True)
    N=len(A)

    degree=np.sum(abs(A),axis=1)
    scipy.io.savemat("./Data/wiki-election.mat",{'degree':degree})
    deg_pos=np.count_nonzero(A==1,axis=1)
    deg_neg=np.count_nonzero(A==-1,axis=1)
    alpha=np.corrcoef(deg_pos,deg_neg)
    print(alpha)
    # The below function was run and the found values are now hardcoded
    # T=calculate_triangles(A)
    T=[8009,55942,135668,407660,607279]
    print(T)
    epsilon=(T[0]+T[2])/T[4]
    epsilon*=0.21
    print("epsilon: "+str(epsilon))
    
    #Matrix coefficeints
    a11=1
    a12=a11*alpha[1][0]
    a22=a11*(np.sum(deg_neg)/np.sum(deg_pos))
    a21=a22*alpha[1][0]
    print("a11: "+str(a11))
    print("a12: "+str(a12))
    print("a21: "+str(a21))
    print("a22: "+str(a22))
    
    # Obtained from running the matlab function 
    # [gamma,xmin,L]=plfit(degree)
    Gamma=2.05
    print("Gamma: "+str(Gamma))
    
    #Delta-Plus and DeltaMinus
    dPlus=(T[3])/(T[3]+T[1])
    dMinus=1-dPlus
    print("Delta Plus: "+str(dPlus))
    print("Delta Minus: "+str(dMinus))
    
    
    beta=0
    beta1=0
    p=0
    p1=0
    Error=N
    # Caculating the values of beta and p
    # That give the least error 
    for i in range(1,100):
        beta=beta+(1/100)
        for j in range(100):
            p=p+(1/100)
            c1=0.5
            c2=dMinus/dPlus
            c11= (beta*a11)/2 + (1-beta)*p*dPlus + (epsilon*p)/(1-epsilon)
            c12= (beta*a12)/2 + (1-beta)*p*dPlus*c2 + (epsilon*p)/(1-epsilon)
            c21= (beta*a21)/2 + (1-beta)*p*dPlus*c1*c2 + (c2*2*epsilon*p)/(1-epsilon)
            c22= (beta*a22)/2 + (1-beta)*p*dPlus*c1
            error=(c11 - (1/(Gamma - 1)))*(c22 - (1/(Gamma - 1))) - c12*c21
            if(Error > abs(error)):
                Error=abs(error)
                p1=p
                beta1=beta
    
    p1=0.67
    beta1=0.13
    print("p: "+str(p1))
    print("beta: "+str(beta1))
    
    
    # Degree Distribution Probability
    D_DistN=np.bincount(degree)
    max_degree=np.max(degree)
    D_DistN=D_DistN/np.sum(D_DistN)
    
    #Cumulative Degree Distribution (P(X>=x))
    cumdegree=np.zeros((max_degree,1))
    for i in range(max_degree):
        cumdegree[i]=np.sum(D_DistN[i:])
    index=list(range(1,max_degree+1))
  
    fig, ax = plt.subplots()
    ax.scatter(index, cumdegree, facecolors='none', edgecolors='black')
    ax.set_xscale("log")
    ax.set_yscale("log");
    plt.xlabel("x (Degree)")
    plt.ylabel("P(X>=x))")
    plt.legend(["Wiki-Election"])
    plt.title("Cumulative Degree Distribution")
    plt.show()

if __name__ == "__main__":
    main()