import numpy as np
import csv
 
A=[]
N=0

def preprocess_file_Bitcoin_Aplha():
    global A
    global N
    file = open("./Data/soc-sign-bitcoinalpha.csv","r")
    csv_reader = csv.reader(file, delimiter=',')
    nodes=0
    for row in csv_reader:
        nodes=max(nodes,max(int(row[0]),int(row[1])))
    N=nodes
    A=np.zeros((nodes,nodes),dtype=int)
    file1 = open("./Data/soc-sign-bitcoinalpha.csv","r")
    csv_reader = csv.reader(file1, delimiter=',')
    for row in csv_reader:
        fr=int(row[0])-1
        to=int(row[1])-1
        value=int(row[2])
        if(value>0):
            A[fr][to]=1
            A[to][fr]=1
        elif(value<0):
            A[fr][to]=-1
            A[to][fr]=-1
            
    A = A[~np.all(A == 0, axis=1)]
    idx = np.argwhere(np.all(A[..., :] == 0, axis=0))
    A=np.delete(A, idx, axis=1)
    print(A.shape)
    print(A)
    N=A.shape[0]

def preprocess_file_Bitcoin_OTC():
	global A
	global N
	
	file = open("./Data/soc-sign-bitcoinotc.csv","r")
	csv_reader = csv.reader(file, delimiter=',')
	nodes=0
	for row in csv_reader:
		nodes=max(nodes,max(int(row[0]),int(row[1])))
	
	N=nodes

	A=np.zeros((nodes,nodes),dtype=int)
	file1 = open("./Data/soc-sign-bitcoinotc.csv","r")
	csv_reader = csv.reader(file1, delimiter=',')
	for row in csv_reader:
		fr=int(row[0])-1
		to=int(row[1])-1
		value=int(row[2])
		if(value>0):
			A[fr][to]=1
			A[to][fr]=1
		elif(value<0):
			A[fr][to]=-1
			A[to][fr]=-1
	
	A = A[~np.all(A == 0, axis=1)]
	idx = np.argwhere(np.all(A[..., :] == 0, axis=0))
	A=np.delete(A, idx, axis=1)
	print(A.shape)
	print(A)
	N=A.shape[0]

def preprocess_file_wiki_election():
    global A
    global N
    file = open("./Data/elec.csv","r")
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        A.append(row)
    print(len(A))
    print(len(A[0]))
    A=np.array(A)
    print(A)
    N=len(A)
    
def main():
	preprocess_file_Bitcoin_OTC()
	with open('./Data/soc-sign-bitcoinotc.npy', 'wb') as f:
		np.save(f, A)
	
	preprocess_file_Bitcoin_Aplha()
	with open('./Data/soc-sign-bitcoinalpha.npy', 'wb') as f2:
		np.save(f2, A)
	
	preprocess_file_wiki_election()
	with open('./Data/wiki-election.npy', 'wb') as f2:
		np.save(f2, A)

if __name__=="__main__":
    main()