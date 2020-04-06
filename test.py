import numpy as numpy
import json

path="/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/Go_game_player/"

def write_moves(result,fpath= path + "moves.txt"):
    with open(fpath,'w') as output:
        output.write(result)

write_moves("0")
f = open (path + 'history_states.txt','w').close()
j = open (path + 'train_set.json','w').close()
j = open (path + 'cresult.txt','w').close()




        
    
    


