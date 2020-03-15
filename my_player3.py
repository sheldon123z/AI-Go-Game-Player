import random
import sys
import numpy as np
from ast import literal_eval as make_tuple
import timeit
import copy
import json
import os

draw_reward = 0
win_reward = 100
lose_reward = -100

boardSize = 5

blank_space = 0
black = 1
white = 2
IN_PLAY = 1
DRAW = 0
WIN = 1
LOSE = -1

def readInput(path = "/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/input.txt"):
    with open(path,'r') as input:
        lines = input.readlines()
        # black or white represents 1 or 2
        side = int(lines[0])

        last_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:6]]
        current_board = [[int(x) for x in line.rstrip('\n')] for line in lines[6:11]]
        
        return side, last_board, current_board

def read_moves(path="/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/moves.txt"):
    with open (path,'r') as input:
        lines = input.readlines()
        current_moves = int(lines[0])
        return current_moves 

def writeOutput(result, path = "/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/output.txt"):
    with open(path,'w') as output:
        if(result == "PASS"):
            output.write("PASS")
        else:
            output.write(''+ str(result[0]) + ',' + str(result[1]))


def write_moves(result,path="/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/moves.txt"):
    with open(path,'w') as output:
        output.write(result)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)       

class Board:
    def __init__(self, size = 5):

        self.size = size
        self.num_moves = 0
        self.max_moves = 12
        self.game_state = IN_PLAY
        self.dead_pieces = []
        self.komi = size/2
        self.me_passed_move = False
        self.op_passed_move = False
        self.capture_reward = 0

    def set_board(self, side, previous_board, current_board):
        self.side = side 
        self.current_board = current_board
        self.previous_board = previous_board

    def set_passed_step(self,side):
        if side == self.side:
            self.me_passed_move = True
        else:
            self.op_passed_move = True

    def if_first_play(self):
        return self.side == black

    def get_dead_opponent_pieces(self):
        dead_pieces =[]
        for row in range(self.size):
            for col in range(self.size):
                if not self.has_liberty(row,col) and current_board[row][col] == 3-self.side:
                    dead_pieces.append(all_allies(row,col))
                    self.dead_pieces = dead_pieces
        return dead_pieces

    def has_dead_pieces(self):
        return self.dead_pieces is not None

    def remove_dead_pieces(self):
        for dead in self.dead_pieces:
            self.current_board[dead[0]][dead[1]] = blank_space  
            #every time eliminate an opponent the capture reward+1
            self.capture_reward += 1
        self.dead_pieces = []

    def current_score_without_komi(self):
        my_score = 0
        op_score = 0
        for row in range(boardSize):
            for col in range(boardSize):
                if self.current_board[row][col]==self.side:
                    my_score+=1
                elif self.current_board[row][col]== (3-self.side):
                    op_score+=1
        return my_score,op_score


    def get_current_score_with_komi(self):
        my_score = 0
        op_score = 0
        for row in range(boardSize):
            for col in range(boardSize):
                if self.current_board[row][col]==self.side:
                    my_score+=1
                elif self.current_board[row][col]== (3-self.side):
                    op_score+=1
        if self.if_first_play():
            my_score += self.komi
        else:
            op_score += self.komi
        return my_score, op_score
        

    def update_game_state(self):
        #remove dead pieces
        if self.has_dead_pieces():
            self.remove_dead_pieces()
        status = self.check_game_status()
        self.game_state = status

        
    def check_game_status(self):
        my_score, op_score = self.current_score_without_komi()
        result = IN_PLAY
        if self.side == white:
            my_score += self.komi
        else:
            op_score += self.komi

        #judge if the game is ended
        if not self.is_game_ended():
            return IN_PLAY
        else:   
            if my_score > op_score:
                result = WIN
            elif my_score < op_score:
                result = LOSE
            else:
                result = DRAW
        return result

    def is_same_board(self):
        
        for row in range(self.size):
            for col in range(self.size):
                if self.current_board[row][col] != self.previous_board[row][col]:
                    return False
        return True
    def get_board_diff_step(self):
        
        if not self.is_same_board():
            for row in range(self.size):
                for col in range(self.size):
                    if self.current_board[row][col] != self.previous_board[row][col]:
                        return row,col
        else:
            return -1,-1



    def is_start_of_game(self):
        chess_count = 0
        chess_color = 0
        for row in range(self.size):
            for col in range(self.size):
                if self.previous_board[row][col] != 0:
                    chess_count+=1
                    chess_color = self.previous_board[row][col]
        if chess_count == 0:
            return True
        if chess_count == 1 and chess_color != self.side:
            return True
        return False
    
    def is_game_ended(self):
        #exceeded the maximum moevs
        self.num_moves = read_moves()

        if self.num_moves >= self.max_moves:
            return True
        elif self.op_passed_move and self.me_passed_move:
            return True
        return False

    def increase_move(self):
        current_moves = read_moves()
        current_moves+=1
        write_moves(str(current_moves))

    def place_stone(self, row, col):
        #place a stone
        self.current_board[row][col]=self.side 

        #update the current moves number of our side
        self.increase_move()

        #check if has dead_piece and update the board 
        self.update_game_state()

        #check if the move create capture reward
        if self.capture_reward != 0:
            reward = self.capture_reward
            self.capture_reward = 0
            return reward
            
    def find_neighbors(self,row,col):
        neighbors = []
        if row > 0:
            neighbors.append((row-1,col))
        if col > 0:
            neighbors.append((row,col-1))
        if row < boardSize-1:
            neighbors.append((row+1,col))
        if col < boardSize-1:
            neighbors.append((row,col+1))
        return neighbors
    
    def all_allies(self, row, col):
        #a queue for dfs
        dfs_stack = [(row,col)]
        allies =[]
        while dfs_stack:
            ally = dfs_stack.pop()
            allies.append(ally)
            neighbors = self.find_neighbors(ally[0],ally[1])
            neighbor_allies = []
            for neighbor in neighbors:
                if neighbor[0]== ally[0] and neighbor[1] == ally[1]:
                    neighbor_allies.append(neighbor)
            for neighbor_allay in neighbor_allies:
                if neighbor_allay not in allies and neighbor_allay not in dfs_stack:
                    dfs_stack.append(neighbor_allay)
        return allies
    
     #check if the move is valid
    def is_valid_place(self,row,col):
        if not self.isOccupied(row,col) or not self.isInBound(row,col):
            return False
        if not self.has_liberty(row,col):
            return False
        else:
            if self.has_dead_pieces():
                self.update_game_state()
            if self.is_violate_ko_rule(row,col):
                return False
            if self.is_an_eye(row,col):
                return False
        return True

    def legal_moves(self):
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if(self.is_valid_place(row,col)):
                    moves.append((row,col))
        return moves


    def isOccupied(self,row,col):
        return self.current_board[row][col] == blank_space
        
    def isInBound(self,row,col):
        return row < boardSize and col < boardSize and row >=0 and col >= 0
        
    def is_violate_ko_rule(self,row,col):

        test = copy.deepcopy(self)
        test.current_board[row][col] = self.side
        test.remove_dead_pieces()
        if(test.has_liberty(row,col)):
            return False
        flag = True
        for col in range(boardSize):
            for row in range(boardSize):
                if test.current_board[row][col] != self.previous_board[row][col]:
                    flag = False
        return flag 

    def is_an_eye(self,row,col):
        if not self.isInBound(row,col):
            return False
        #检测周围的棋子都是己方棋子
        neighbors = self.find_neighbors(row,col)
        for neighbor in neighbors:
            if self.current_board[neighbor[0]][neighbor[1]] != self.side:
                return False
        
        #得到4个角的坐标
        corners = [(row-1,col-1),(row+1,col+1),(row-1,col+1),(row+1,col-1)]
        friend_corners = 0
        out_of_board_corners = 0

        #检测四个角的棋子至少有三个是己方棋子
        for point in corners:
            if self.isInBound(point[0],point[1]) and self.current_board[point[0]][point[1]]==self.side:
                friend_corners += 1
            else:
                out_of_board_corners += 1
        if out_of_board_corners > 0:
            if friend_corners + out_of_board_corners == 4:
                return True
        return friend_corners >= 3

    def liberty_count(self,row,col):
        count=0
        allies = self.all_allies(row,col)
        for ally in allies:
            neighbors = ally.find_neighbors()
            for neighbor in neighbors:
                if self.current_board[neighbor[0]][neighbor[1]]==0:
                    count+=1
        return count

    def has_liberty(self,row,col):
        allies = self.all_allies(row,col)
        for ally in allies:
            neighbors = self.find_neighbors(ally[0],ally[1])
            for neighbor in neighbors:
                if self.current_board[neighbor[0]][neighbor[1]] == blank_space:
                    return True
        return False

        
class Qplayer:
    def __init__(self, board = None, side = None, alpha = 0.5, gamma = 0.8 ,initial_value = 0.5):
        self.board = board
        self.side = side
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self.history_states = []
        self.initial_value = initial_value

    def init_q_values(self):
        with open ('/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/train_set.json','r+') as f:
            # lines = f.readlines()
            # for line in lines:
            try:
                data = json.load(f)
                self.q_values.update(data)
            except:
                print("no q values in json file")
        
    def Q(self, state):
        if state not in self.q_values:
            q_val = np.zeros((5,5))
            q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    def get_best_move(self,board):
        legal_moves = board.legal_moves()
        #如果没有legal move就返回pass
        if not(legal_moves):
            return -1, -1

        state = self.encode_board(board.side, board.current_board,legal_moves[0],legal_moves[1])

        q_values = self.Q(state)
        # print(q_values,type(q_values))
        #选择一个在legal move里面并且是最大的q值的坐标点
        while True:
            point = self.find_max_coord(q_values)
            if point in legal_moves:
                return point[0],point[1]
            #如果没有在legal move 就说明当前点不可以选，更新这个点的qvalue为-1，即在当前棋面下这个点永远不走
            else:
                q_values[point[0]][point[1]] = -1

    def find_max_coord(self,q):
        max_q = -np.inf
        row = 0
        col = 0
        for i in range(boardSize):
            for j in range(boardSize):
                if q[i][j] > max_q:
                    max_q = q[i][j]
                    row, col = i, j
        return (row, col)

    def make_one_move(self,row,col):
        if row == col == -1:
            self.board.set_passed_step(self.side)
            return "PASS"
        else:
            self.board.place_stone(row,col)
            return row,col
            

    def update_learned(self):
        with open('/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/train_set.json','r+') as json_file:
            try:
                dic = json.load(json_file)
                dic.update(self.q_values)
                json_file.seek(0)
                json.dump(dic,json_file,cls = NpEncoder)
            except:
                json.dump(self.q_values,json_file,cls = NpEncoder)
                




    def save_states(self,state):
        with open ('/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/history_states.txt','a+') as file:
            file.seek(0)
            data = file.read()
            if len(data)>0:
                file.write("\n")
            file.write(state)

    def read_states(self):
        with open ('/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/history_states.txt','r+') as file:
            lines = file.readlines()
            return lines

    def encode_board(self, side, board, row, col):
          #2D array
        board_matrix = np.zeros((boardSize,boardSize),dtype=int)
        #self is 1 opponent is -1
        for r in range(boardSize):
            for c in range(boardSize):
                if board[r][c] == side:
                    board_matrix[r][c] = 1
                elif board[r][c] == blank_space:
                    board_matrix[r][c] = 0
                else:
                    board_matrix[r][c] = -1

        state = ''.join([str(board_matrix[j][i])+" " for j in range(boardSize) for i in range(boardSize)]) +"#"+ str(row)+ "#" + str(col)
        
        return state

    def decode_state(self,state):
        board_matrix = np.zeros((boardSize,boardSize))
        print(state)
        arr = state.split(" ")
        data = []
        for n in arr:
            data.append(int(n))
        
        for i in range(boardSize):
            for j in range(boardSize):
                if len(data) > 0:
                    board_matrix[j][i] = data.pop(0)
        #transpose the array list
        l2 = np.transpose(board_matrix)
        #print(l2,type(l2))
        return l2

    def get_state_reward(self,state):
        arr = state.split(" ")
        # board_arr = self.decode_state(state)
        score = 0
        for num in arr:
            if num:
                score += int(num)
        # for row in range(len(board_arr)):
        #     for col in range(len(board_arr[0])):
        #         score += board_arr[row][col]
        return score

        
    def train_after_end(self,result):
        if result == DRAW:
            result_reward = draw_reward
        else:
            if result == WIN:
                result_reward = win_reward
            if result == LOSE:
                result_reward = lose_reward
        self.history_states.reverse()
        max_q = -np.inf
        for hist in self.history_states:
            l = hist.split("#")
            #get the board state information
            board_state = l[0]
            #get the step coordinate 落子坐标
            row = int(l[1])
            col = int(l[2])
            
            #获取状态奖励
            state_reward = self.get_state_reward(board_state)
            # print("state {} state reward {}\n".format(board_state,state_reward))
            
            #获取q value matrix
            q = self.Q(hist)
            
            #print("q value: {}".format(q),type(q))

            #如果是最后一步的状态，即最大值初设置为-
            if max_q< 0:
                q[row][col] = result_reward
            else:
                q[row][col] = q[row][col] * (1 - self.alpha) + self.alpha * (state_reward + self.gamma * max_q)
            max_q = np.max(q)
        
        self.update_learned()

    def clear_history_file(self, path ="/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/history_states.txt"):
        file = open(path,'w').close()
        
            
    #set moves to 0
    def clean_up_after_end(self,result):
        if not board.is_game_ended():
            return
        #set 0 to moves
        write_moves("0")
        #clear history states file
        self.clear_history_file()

    def play(self):
        if(self.board.is_start_of_game()):
            write_moves("0")
            row, col = self.get_best_move(self.board)
            action = self.make_one_move(row, col)
            print("current_moves: {}".format(self.board.num_moves))
            state = self.encode_board(side, self.oard.current_board,row,col)
            self.save_states(state)
        else:
            if self.board.is_same_board():
                self.board.op_passed_move = True
            #read the history states 
            states = self.read_states()
            for state in states:
                self.history_states.append(state)
            
            #read q values from file
            self.init_q_values()
            #get the best move
            action = self.get_best_move(self.board)
            #if the action is pass then 
            if action[0] == -1 and action[1] == -1:
                self.board.set_passed_step(self.side)
                self.board.increase_move()
                self.board.num_moves = read_moves()
                print("current_moves: {} this move is PASS".format(self.board.num_moves))
                action = "PASS"
                
                #check if game is ended
                if self.board.is_game_ended():
                    result = self.board.check_game_status()
                    self.train_after_end(result)
                    self.clean_up_after_end(result)
                    print("predicted result: {}".format(result))
            else:
                #make a move
                action = self.make_one_move(action[0],action[1])
                print("current_moves: {} this move {}".format(self.board.num_moves,(action[0],action[1])))
                state = self.encode_board(side, self.board.current_board,action[0],action[1])
                #save state to file and append the state to history states
                self.save_states(state)
                self.history_states.append(state)
                
                #check if game is ended after the move,train if game is ended
                if self.board.is_game_ended():
                    result = self.board.check_game_status()
                    self.train_after_end(result)
                    self.clean_up_after_end(result)
                    print("predicted result: {}".format(result)) 

# class MinmaxPlayer:
#     def __init__(self):
        

if __name__ == "__main__":

    #initialize the board 
    side, last_board, current_board = readInput()
    board = Board(boardSize)
    board.set_board(side,last_board, current_board)
    #create a player
    player = Qplayer(board,side)
    
    #if this is the start of the game
    if(board.is_start_of_game()):
        write_moves("0")
        row, col = player.get_best_move(board)
        action = player.make_one_move(row, col)
        print("current_moves: {}".format(board.num_moves))
        state = player.encode_board(board.side, board.current_board,row,col)
        player.save_states(state)
    else:
        if board.is_same_board():
            board.op_passed_move = True
        #read the history states 
        states = player.read_states()
        for state in states:
            player.history_states.append(state)
        
        #read q values from file
        player.init_q_values()
        #get the best move
        action = player.get_best_move(board)
        #if the action is pass then 
        if action[0] == -1 and action[1] == -1:
            board.set_passed_step(player.side)
            board.increase_move()
            board.num_moves = read_moves()
            print("current_moves: {} this move is PASS".format(board.num_moves))
            action = "PASS"
            
            #check if game is ended
            if board.is_game_ended():
                result = board.check_game_status()
                player.train_after_end(result)
                player.clean_up_after_end(result)
                print("predicted result: {}".format(result))
        else:
            #make a move
            action = player.make_one_move(action[0],action[1])
            print("current_moves: {} this move {}".format(board.num_moves,(action[0],action[1])))
            state = player.encode_board(board.side, board.current_board,action[0],action[1])
            #save state to file and append the state to history states
            player.save_states(state)
            player.history_states.append(state)
            
            #check if game is ended after the move,train if game is ended
            if board.is_game_ended():
                result = board.check_game_status()
                player.train_after_end(result)
                player.clean_up_after_end(result)
                print("predicted result: {}".format(result))

        #output the move
    writeOutput(action)

 
    




