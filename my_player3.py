import random
import sys
import numpy as np
import timeit
import copy

draw_reward = 0
win_reward = 100
lose_reward = -100

boardSize = 5
max_moves = boardSize * boardSize - 1
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

def writeOutput(result, path = "/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/output.txt"):
    with open(path,'w') as output:
        if(result == "PASS"):
            output.write("PASS")
        else:
            output.write(''+ str(result[0]) + ',' + str(result[1]))
        
class Encoder():
    def  __init__(self, board_size):
        
        self.board_width = board_size
        self.board_height = board_size
    

    def encode_board(self, board):
      #2D array
      board_matrix = np.zeros((boardSize,boardSize))
      side = board.side
      #self is 1 opponent is -1
      for r in range(self.board_width):
          for c in range(self.board_width):
              if board.current_board[r][c] == side:
                  board_matrix[r, c] = 1
              else:
                  board_matrix[r, c] = -1
          #set the ko point as 1 in layer 2
            #   if board.is_violate_ko_rule(r,c):
            #       board_matrix[1,r,c] = 1
      state = ''.join([str(board_matrix[j][i])+' ' for j in range(boardSize) for i in range(boardSize)])
      #print(state,type(state))
      return state
    def decode_state(self,state):
        board_matrix = np.zeros((boardSize,boardSize))
        l = state.split()
        #print(l)
        data = []
        for n in l:
            data.append(int(float(n)))

        for i in range(boardSize):
            for j in range(boardSize):
                board_matrix[j][i] = data.pop(0)

        #transpose the array list
        l2 = np.transpose(board_matrix)
        #print(l2,type(l2))
        return l2
    
    def encode_point(self,row, col):
      #将坐标点转换为整数索引
      return self.board_width * row + col
    
    def decode_point_index(self, code):
      row = code // self.board_width
      col = code % self.board_width
      return (row, col)
    
    def num_points(self):
      return self.board_width * self.board_height
    
    def shape(self):
      return self.board_height, self.board_width

class Board:
    def __init__(self, side, previous_board, current_board, size = 5):

        #side choose
        self.side = side 
        self.current_board = current_board
        self.previous_board = previous_board
        self.size = size
        self.num_moves = 0
        self.max_moves = size * size -1
        self.game_state = IN_PLAY
        self.dead_pieces = []
        self.komi = size/2
        self.capture_reward = 0
    
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
        status = self.judge()
        self.game_state = status

        
    def judge(self):
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
            if my_score >= op_score:
                result = WIN
            elif my_score <= op_score:
                result = LOSE
            else:
                result = DRAW
        return result

    def is_game_ended(self):
        #exceeded the maximum moevs
        if self.num_moves >= self.size * self.size - 1:
            return True
        return False

    def place_stone(self, row, col):
        #if the position is valid to place a stone then place a stone
        if self.is_valid_place(row,col):
            #place a stone
            self.current_board[row][col]=self.side 
            #先增加1步
            self.num_moves += 1
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
        if not self.isOccupied(row,col):
            return False
        if not self.isInBound(row,col):
            return False
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
            neighbors = self.find_neighbors(row,col)
            for neighbor in neighbors:
                if self.current_board[row][col] == blank_space:
                    return True
        return False

        
class Qplayer:
    def __init__(self, encoder, board = None, side = None, alpha = 0.5, gamma = 0.8 ,initial_value = 0.5):
        self.board = board
        self.side = side
        self.alpha = alpha
        self.gamma = gamma
        self.encoder = encoder
        self.q_values = {}
        self.history_states = []
        self.initial_value = initial_value

    def Q(self, state):
        if state not in self.q_values:
            q_val = np.zeros((5,5))
            q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    def get_best_move(self):
        state = self.encoder.encode_board(self.board)
        legal_moves = self.board.legal_moves()
        #如果没有legal move就返回pass
        if not(legal_moves):
            return -1, -1

        q_values = self.Q(state)
        #选择一个在legal move里面并且是最大的q值的坐标点
        while True:
            point = np.unravel_index(q_values.argmax(), q_values.shape)
            if point in legal_moves:
                return point[0],point[1]
            #如果没有在legal move 就说明当前点不可以选，更新这个点的qvalue为-1，即在当前棋面下这个点永远不走
            else:
                q_values[point[0]][point[1]] = -1

    
    def make_one_move(self):
        if self.board.is_game_ended():
            return
        row, col = self.get_best_move()
        self.history_states.append((self.encoder.encode_board(self.board),(row,col)))
        
        if row == col == -1:
            return "PASS"
        else:
            reward = self.board.place_stone(row,col)
            return (row,col)

    def get_state_reward(self,state):
        board_arr = self.encoder.decode_state(state)
        score = 0
        for row in range(len(board_arr)):
            for col in range(len(board_arr[0])):
                score += board_arr[row][col]
        return score
        
    def train_after_end(self,result):
        if result == DRAW:
            result_reward = draw_reward
        else:
            my_score, op_score = self.board.get_current_score_with_komi()
            if result == WIN:
                result_reward = win_reward
            if result == LOSE:
                result_reward = lose_reward
        self.history_states.reverse()
        max_q = -1
        for hist in self.history_states:
            state, move = hist
            #获取q value matrix
            q = self.Q(state)

            state_reward = self.get_state_reward(state)

            if max_q< 0:
                q[move[0]][move[1]] = result_reward
            else:
                q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha * (state_reward + self.gamma * max_q)
            max_q = np.max(q)
        self.history_states = []

# class MinmaxPlayer:
#     def __init__(self):
        

if __name__ == "__main__":
    time_limit = 300
    side, last_board, current_board = readInput()
    board = Board(side,last_board, current_board, boardSize)
    encoder = Encoder(boardSize)
    player = Qplayer(encoder,board,side)
    action = player.make_one_move()
    
    game_state = player.board.game_state
    #output the move
    writeOutput(action)
    if game_state == WIN or game_state == LOSE:
        player.train_after_end(game_state)




