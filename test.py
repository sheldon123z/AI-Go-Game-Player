import numpy as numpy
import json

def write_moves(result,path="/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/Go_game_player/moves.txt"):
    with open(path,'w') as output:
        output.write(result)

write_moves("0")
f = open ('/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/Go_game_player/history_states.txt','w').close()
j = open ('/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/Go_game_player/train_set.json','w').close()





        
    
    


