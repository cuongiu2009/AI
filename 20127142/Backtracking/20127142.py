import time
import sys
start=time.time()
n=int(sys.argv[6])
x=int(sys.argv[4])-1
y=int(sys.argv[2])-1
chess_board = [[-1 for i in range(n)]for i in range(n)] 
countMove=0
writed=False
f=open('20127142_backtrack.txt','w+')
f.write(str(y+1)+' '+str(x+1)+' '+str(n)+'\n')
def isValid(x,y,chess_board): 
    if(x >= 0 and y >= 0 and x < n and y < n and chess_board[x][y] == -1): 
        return True
    return False

def writeFile(chess_board,n,runTime): 
    f.write(str(countMove)+'\n')
    f.write(str(runTime*1000)+'\n')
    
    for i in range(n): 
      for j in range(n): 
        f.write(str(chess_board[i][j]))
        if (j!=n-1):
          f.write(' ') 
      if (i!=n-1):
        f.write('\n')
    


def backtrack(n,x,y): 
    pos_x = [2, 1, -1, -2, -2, -1, 1, 2] 
    pos_y = [1, 2, 2, 1, -1, -2, -2, -1] 
    

    chess_board[x][y] = 1
    
    pos = 2
    
    if(not backTrackTour(n,chess_board, x, y, pos_x, pos_y, pos)): 
      return False
    else:
      global end
      end=time.time() 
      writeFile(chess_board,n,end-start) 

def backTrackTour(n,chess_board,curr_x,curr_y,pos_x,pos_y,pos): 
    global writed
    if (writed==True):
      return False
    end=time.time()
    if (end-start>3600):
      writed=True
      writeFile(chess_board,n,end-start)
      return False
    if(pos == n**2+1): 
      return True
    
  
    for i in range(8): 
      new_x = curr_x + pos_x[i] 
      new_y = curr_y + pos_y[i] 

      if(isValid(new_x,new_y,chess_board)): 
        global countMove
        countMove=countMove+1
        chess_board[new_x][new_y] = pos 
        if(backTrackTour(n,chess_board,new_x,new_y,pos_x,pos_y,pos+1)):
            return True

        chess_board[new_x][new_y] = -1
    return False
      
backtrack(n,x,y)