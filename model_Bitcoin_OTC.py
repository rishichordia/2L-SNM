import matplotlib.pyplot as plt
import numpy as np
import random
def getColumns(Mat,j):
    ans=[]
    N=len(Mat)
    for i in range(N):
      if(Mat[i][j]==1):
        ans.append(i)
    return ans
def model_2LSNM(A):
    N=len(A)
    N=100
    T=[348,5691,4434,23020,33493]
    a11=1
    a12=0.43403524775339286
    a21=0.07710368620364381
    a22=0.17764383561643834
    d_p=0.8017832886350179
    epsilon=0.19*0.14277610246917266
    p=0.63
    Beta=0.38
    
    L1=np.zeros((N,N),dtype=int)
    L2=np.zeros((N,N),dtype=int)
    i=4;
    L1[1][2]=L1[2][1]=1
    L1[3][2]=L1[2][3]=1
    L1[4][2]=L1[2][4]=1
    L1[1][3]=L1[3][1]=1
    L1[4][3]=L1[3][4]=1    
    L2[1][4]=L2[4][1]=1
    
    degree=np.sum(abs(A),axis=1)
    leaf_freq=len(np.argwhere(degree==1))
    while(i<N):
        print(i)
        growth_allowed=True
        pPlus=p
        beta=Beta
        dPlus=np.sum(L1,axis=1)
        dMinus=np.sum(L2,axis=1)
        # deltaP=0.5  
        deltaP=T[3]/(T[1]+T[3])
        # deltaN=0.5 
        deltaN=1-deltaP
        if(i > N-leaf_freq):
            pPlus=0
            beta=0
            growth_allowed=False
            p1=np.add(a11*dPlus,a12*dMinus)
            p1=p1/i
            p2=np.add(a21*dPlus,a22*dMinus)
            p2=p2/i
            
            i+=1
            failed=1
            while(failed):
                for j in range(i-1):
                    if((random.random()<=p1[j]) & (L2[i-1][j]==0)):
                        failed=0
                        L1[i-1][j]=1
                        L1[j][i-1]=1
                        break
            continue
         
        pMinus=p/2
        if(random.random()<=(1-epsilon)): # A new node will appear
            # Process-I
            if(random.random()<=beta):

                p1=np.add(a11*dPlus,a12*dMinus)
                p1=p1/i
                p2=np.add(a21*dPlus,a22*dMinus)
                p2=p2/i
                i+=1
                success=0
                while(success!=1):
                    for j in range(i-1):
                        rr=random.random()
                        if(rr <=0.5):
                            #Positive Link
                            rrr=random.random()
                            if((rrr<=p1[j]) & (L2[i-1][j]==0)):
                                success=1
                                L1[i-1][j]=L1[j][i-1]=1
                        else:
                            #negative link
                            if((random.random()<=p2[j]) & (L1[i-1][j]==0)):
                                success=1
                                L2[i-1][j]=L2[j][i-1]=1
            
            ##Process II
            else:
                i=i+1
                j=random.randint(0,i-2)
                Nb1=getColumns(L1,j)
                Nb2 = getColumns(L2,j)
                if(random.random()<=deltaN):
                    L2[i-1][j]=1
                    L2[j][i-1]=1
                    for k in range(len(Nb1)): #-+-
                        if((random.random()<=pMinus) & (L1[i-1][Nb1[k]]==0)):
                            L2[i-1][Nb1[k]]=1
                            L2[Nb1[k]][i-1]=1

                    for k in range(len(Nb2)):#--
                        if((random.random()<=pPlus) & L2[i-1][Nb2[k]]==0):
                            L1[i-1][Nb2[k]]=1
                            L1[Nb2[k]][i-1]=1
                else:
                    L1[i-1][j]=1
                    L1[j][i-1]=1
                    for k in range(len(Nb1)): #+++
                        if((random.random()<=pPlus) & (L2[i-1][Nb1[k]]==0)):
                            L1[i-1][Nb1[k]]=1
                            L1[Nb1[k]][i-1]=1

                    for k in range(len(Nb2)):#+--
                        if((random.random()<=pMinus) & (L1[i-1][Nb2[k]]==0)):
                            L2[i-1][Nb2[k]]=1
                            L2[Nb2[k]][i-1]=1
        else:
            #Process III - Internal Growth Process (No new node will appear)
            x=random.randint(0,i-2)
            Nb1=getColumns(L1,x)
            Nb2=getColumns(L2,x)
            if(random.random()<=0.5):
                #Positive Layer
                for k in range(len(Nb1)):   #++-
                    Nbb1=getColumns(L1,Nb1[k])
                    if not Nbb1:
                        for y in range(len(Nbb1)):
                            if((random.random()<pMinus) & (L1[x][Nbb1[y]]==0) & (L2[x][Nbb1[y]]==0)):
                                L2[x][Nbb1[y]]=1
                                L2[Nbb1[y]][x]=1
                                break;
                
                for k in range(len(Nb1)):  #+-+
                    Nbb2=getColumns(L2,Nb1[k])
                    if not Nbb2:
                        for y in range(len(Nbb2)):
                            if((random.random()<pPlus) & (L2[x][Nbb2[y]]==0) & (L2[x][Nbb2[y]]==0)):
                                L1[x][Nbb1[y]]=1
                                L1[Nbb1[y]][x]=1
                                break;
            else:
                #Negative Layer
                for k in range(len(Nb2)):   #-++
                    Nbb1=getColumns(L1,Nb2[k])
                    if not Nbb1:
                        for y in range(len(Nbb1)):
                            if((random.random()<pPlus) & (L1[x][Nbb1[y]]==0) & (L2[x][Nbb1[y]]==0)):
                                L1[x][Nbb1[y]]=1
                                L1[Nbb1[y]][x]=1
                                break;
                
                for k in range(len(Nb2)):   #---
                    Nbb2=getColumns(L2,Nb2[k])
                    if not Nbb2:
                        for y in range(len(Nbb2)):
                            if((random.random()<pMinus) & (L1[x][Nbb2[y]]==0) & (L2[x][Nbb2[y]]==0)):
                                L2[x][Nbb2[y]]=1
                                L2[Nbb2[y]][x]=1
                                break;
            
    print("Model execution Finished:")
    print(L1)
    print(L2)
    L=np.subtract(L1,L2)
    return L

def plotLogLog(A,ax,m,facecolors='black',edgecolors='face'):
    degree=np.sum(abs(A),axis=1)
    deg_pos=np.count_nonzero(A==1,axis=1)
    deg_neg=np.count_nonzero(A==-1,axis=1)
    
    D_DistN=np.bincount(degree)
    max_degree=np.max(degree)
    D_DistN=D_DistN/np.sum(D_DistN)
    
    cumdegree=np.zeros((max_degree,1))
    for i in range(max_degree):
        cumdegree[i]=np.sum(D_DistN[i:])
    index=list(range(1,max_degree+1))
    
    ax.scatter(index, cumdegree,marker=m,facecolors=facecolors, edgecolors=edgecolors)
    ax.set_xscale("log")
    ax.set_yscale("log");
    return ax

def calculate_Distribution(A):
    degree=np.sum(abs(A),axis=1)
    deg_pos=np.count_nonzero(A==1,axis=1)
    deg_neg=np.count_nonzero(A==-1,axis=1)
    
    D_DistN=np.bincount(degree)
    max_degree=np.max(degree)
    D_DistN=D_DistN/np.sum(D_DistN)
    
    cumdegree=np.zeros((max_degree,1))
    for i in range(max_degree):
        cumdegree[i]=np.sum(D_DistN[i:])
    return (cumdegree,D_DistN)

def calculate_error(X,Y):
    error=0
    if(len(X)<=len(Y)):
      for i in range(len(X)):
        error+=abs((1-X[i])-(1-Y[i]))
      for i in range(len(X),len(Y)):
        error+=Y[i]
    else: 
      return calculate_error(Y,X)
    return error


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

def calculate_performance(G,A):
    print("\n\nPerformance parameters")
    print("Fraction of Triads in real model:")
    T_r=calculate_triangles(A)
    
    f_r=[x/T_r[4] for x in T_r]
    for i in range(4):
        print("T"+str(i)+": " +str(f_r[i]))
    print("Fraction of Triads in generated model:")
    T_g=calculate_triangles(G)
    f_g=[x/T_g[4] for x in T_g]

    error=[abs(f_g[i]-f_r[i]) for i in range(4)]
    total_error=np.sum(error)
    for i in range(4):
        print("T"+str(i)+": " +str(f_g[i])+ " Error:"+str(error[i]))
    print("Absolute Error:"+str(total_error))

    TB=T_g[3]+T_g[1]
    TBr=T_r[3]+T_r[1]
    TUB=T_g[2]+T_g[0]
    TUBr=T_r[2]+T_r[0]
    Csr=(TBr-TUBr)/T_r[4]
    Cs=(TB-TUB)/T_g[4]
    print()
    print("Clustering Coefficient of real graph Cs(G): "+ str(Csr))
    print("Clustering Coefficient 2L-SNM Cs(G): "+ str(Cs))
    Sg=(TB-TUB)/(TB+TUB)
    Sgr=(TBr-TUBr)/(TBr+TUBr)
    print("Relative Signed Clustering Coefficient of real graph S(G): "+ str(Sgr))
    print("Relative Signed Clustering Coefficient of 2L-SNM S(G): "+ str(Sg))

    D_UB=TUB/TB
    D_UBr=TUBr/TBr
    print("Degree of unbalance (Real graph): " + str(D_UBr))
    print("Degree of unbalance (2L-SNM): " + str(D_UB))
    
    
def main():
    with open("./Data/soc-sign-bitcoinotc.npy", 'rb') as file:
        A = np.load(file,allow_pickle=True)
    N=len(A)
    cumdegree , D_DistN =calculate_Distribution(A)
    Generated_Graph=model_2LSNM(A)
    cumdegree_G,D_DistN_G =calculate_Distribution(Generated_Graph)
    
    calculate_performance(Generated_Graph,A)
    
    error=calculate_error(cumdegree,cumdegree_G)
    print("Error: "+str(error))
    fig, ax = plt.subplots()
    ax=plotLogLog(A,ax,'o','None','black')
    ax=plotLogLog(Generated_Graph,ax,'x','red')
    plt.xlabel('x (Degree)')
    plt.ylabel('Pr (X>=x)')
    plt.legend(["Bitcoin-OTC","2L-SNM"])
    plt.title("Degree Distribution Plot")
    name="Bitcoin-OTC.jpg"
    plt.savefig(name)
    plt.show()
                
if __name__=="__main__":
    main()