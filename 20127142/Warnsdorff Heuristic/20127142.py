#Warnsdoff
from ast import Break
import sys
import time
start=time.time()
chess_board=[]
pos_x=(2,1,2,1,-2,-1,-2,-1)
pos_y=(1,2,-1,-2,1,2,-1,-2)
n=int(sys.argv[6])
x=int(sys.argv[4])-1
y=int(sys.argv[2])-1
countMove=0
f=open('20127142_heuristic.txt','w+')
f.write(str(y+1)+' '+str(x+1)+' '+str(n)+'\n')
for i in range(n):
    chess_board.append([])
    for j in range(n):
        chess_board[i].append(-1)
    

def possibilities(x,y,n):
    poss=[]
    global countMove
    for i in range(8):
        if x+pos_x[i]>=0 and x+pos_x[i]<=n-1 and y+pos_y[i]>=0 and y+pos_y[i] <=n-1 and chess_board[x+pos_x[i]][y+pos_y[i]]==-1:
            poss.append([x+pos_x[i],y+pos_y[i]])
            countMove+=1
    
    return poss

def firstMove(x,y):
    chess_board[x][y]=1
    return chess_board
#Warnsdoff algo
count=2
firstMove(x,y)
for i in range(n*n-1):
    possition=possibilities(x,y,n)
    min=possition[0]
    for p in possition:
        if len(possibilities(p[0],p[1],n))<=len(possibilities(min[0],min[1],n)):
            min=p
    x=min[0]
    y=min[1]
    chess_board[x][y]=count
    count+=1
#####
end=time.time()
f.write(str(countMove)+'\n')
f.write(str("{:.10f}".format((end-start)*1000)))
f.write('\n')
for i in range(n):
    for j in range(n):
        f.write(str(chess_board[i][j]))
        if (j!=n-1):
            f.write(' ')
    if (i!=n-1):
        f.write('\n')