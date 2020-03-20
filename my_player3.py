import random
import numpy as np
import time
import copy
import json

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
MIN = -9999
MAX = 9999

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

def increase_move():
    current_moves = read_moves()
    current_moves+=1
    write_moves(str(current_moves))

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

    def has_dead_pieces(self,side):
        dead_pieces = []
        for row in range(boardSize):
            for col in range(boardSize):
                if self.current_board[row][col] == side:
                    if not self.has_liberty(row, col):
                        dead_pieces.append((row,col))
        return dead_pieces

    def remove_dead_pieces(self,side):
        for row in range(self.size):
            for col in range(self.size):
                if not self.has_liberty(row,col) and current_board[row][col] == side:
                    self.current_board[row][col] = blank_space  
                    #every time eliminate an opponent the capture reward+1
                    self.capture_reward += 1

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


    def black_white_score(self):
        black = 0
        white = 0
        for row in range(boardSize):
            for col in range(boardSize):
                if self.current_board[row][col]== 1:
                    black+=1
                elif self.current_board[row][col]== 2:
                    white+=1
        return black, white
        

    def update_game_state(self):
        #remove dead pieces
        self.remove_dead_pieces(3-self.side)
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

        if self.num_moves >= self.max_moves or (self.op_passed_move and self.me_passed_move):
            return True
        return False

    #put a stone on self board
    def place_stone(self, row, col):
        #place a stone
        self.current_board[row][col]=self.side 

        #check if the move create capture reward
        if self.capture_reward != 0:
            reward = self.capture_reward
            self.capture_reward = 0
            return reward

    def make_test_move(self, side, row,col):
        test_board = copy.deepcopy(self)
        test_board.current_board[row][col] = side
        test_board.side = 3-side
        test_board.remove_dead_pieces(3-test_board.side)

        return test_board



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
            neighbor_allies = self.get_neighbor_ally(ally[0],ally[1])
            for neighbor_allay in neighbor_allies:
                if neighbor_allay not in allies and neighbor_allay not in dfs_stack:
                    dfs_stack.append(neighbor_allay)
        return allies
    
    def get_neighbor_ally(self, i, j):
        
        board = self.current_board
        neighbors = self.find_neighbors(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

     #check if the move is valid
    def is_valid_place(self,row,col):
        if not self.isOccupied(row,col) or not self.isInBound(row,col):
            return False
        
        test = copy.deepcopy(self)
        test.current_board[row][col] = test.side
        test.remove_dead_pieces(3-test.side)

        #check if is an eye
        # if self.is_an_eye(row,col):
        #     return False

        if test.has_liberty(row,col):
            return True
        else:
            test.remove_dead_pieces(3-test.side)
            if not test.has_liberty(row,col):
                return False
        #check ko rule
        flag = True
        for row in range(boardSize):
            for col in range(boardSize):
                if test.current_board[row][col] != self.previous_board[row][col]:
                    flag = False
                
        return True and flag

    def legal_moves(self):
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if(self.is_valid_place(row,col)):
                    point = (row,col)
                    moves.append(point)
        return moves


    def isOccupied(self,row,col):
        return self.current_board[row][col] == blank_space
        
    def isInBound(self,row,col):
        return row < boardSize and col < boardSize and row >=0 and col >= 0
        

    def is_acquired_position(self,row,col,side):

        neighbors = self.find_neighbors(row,col)
        for neighbor in neighbors:
            if self.current_board[neighbor[0]][neighbor[1]] != side:
                return False
        return True

    def is_an_eye(self,row,col):
        if not self.isInBound(row,col) or self.isOccupied(row,col):
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
            if self.isInBound(point[0],point[1]): 
                if self.current_board[point[0]][point[1]]==self.side:
                    friend_corners += 1
            else:
                out_of_board_corners += 1
        if out_of_board_corners > 0:
            if friend_corners + out_of_board_corners == 4:
                return True
        return friend_corners >= 3


    def has_liberty(self,row,col):
        flag = False
        allies = self.all_allies(row,col)
        for ally in allies:
            neighbors = self.find_neighbors(ally[0],ally[1])
            for neighbor in neighbors:
                if self.current_board[neighbor[0]][neighbor[1]] == blank_space:
                    return True
        return False

    def alpha_beta_search(self,state,max_depth):

        ans_move = None
        move_dic = dict()
        best_score = np.inf
        possible_moves = state.legal_moves()
        alpha = MIN
        beta = MAX
        value = MIN

        max_dead = 0
        instant_kill_move = None
        current_move = read_moves()
        for move in possible_moves: 

            greedy = copy.deepcopy(self)
            greedy.place_stone(move[0],move[1])
            deads = greedy.has_dead_pieces(3-greedy.side)
            can_kill_num = len(deads)
            print("leng of deads {}".format(can_kill_num))
            if max_dead < can_kill_num:
                max_dead = can_kill_num
                instant_kill_move = move
            if max_dead >=2:
                print("find killing move {}".format(instant_kill_move))
                print("-------------this move is made by greedy----------------")
                # return instant_kill_move

            next_state = state.make_test_move(state.side, move[0],move[1])

            value = Min_value(next_state, alpha, beta, max_depth, evaluation,0)
            move_dic.setdefault(value,[]).append(move)
            print("current move in ab search {} value {}".format(move,value))
            if best_score >= value:
                best_score = value
            else:
                alpha = max(alpha,value)
            #print("best_score{},move{}".format(best_score, move))
        print("-------------this move is made by greedy----------------")
        return random.choice(move_dic[best_score])

def Min_value(state, a, b, max_depth,eva_function,layer):
    
    possible_moves = state.legal_moves()
    if read_moves() + layer >= state.max_moves or len(possible_moves)==0:# or max_depth == 0:
        result = state.check_game_status()
        black_s, white_s = state.black_white_score()
        if result == WIN:
            return -abs(black_s-white_s)-state.komi 
        elif result == LOSE:
            return abs(black_s-white_s)+state.komi
        elif result == DRAW:
            return 0

    elif max_depth == 0 :
        result = eva_function(state)
        return result
    
    value = MAX
    
    for move in possible_moves:
        next_state = state.make_test_move(state.side, move[0],move[1])
        value = min(value, Max_value(next_state,a,b,max_depth-1,eva_function,layer+1))
        if value <= a:
            return value
        b = min(b,value)
    return value

def Max_value(state, a, b, max_depth,eva_function, layer):

    possible_moves = state.legal_moves()
    if read_moves() + layer  >= state.max_moves or len(possible_moves)==0: #or max_depth == 0:
        result = state.check_game_status()
        black_s, white_s = state.black_white_score()
        if result == WIN:
            return  abs(black_s-white_s)+state.komi 
        elif result == LOSE:
            return -abs(black_s-white_s)-state.komi 
        elif result == DRAW:
            return 0
        
    elif max_depth == 0 :
        result = eva_function(state)
        return result
    
    value = MIN
    
    for move in possible_moves:
        next_state = state.make_test_move(state.side, move[0],move[1])
        value = max(value, Min_value(next_state,a,b,max_depth-1,eva_function,layer+1))
        if value >= b:
            return value
        a = max(a,value)
    return value




def evaluation(state):

    black_score = 0
    white_score = 0

    black_liberty = 0
    white_liberty = 0

    black_num = 0
    white_num = 0

    black_eyes = 0
    white_eyes = 0

    black_acquired = 0
    white_acquired = 0

    black_pos = []
    white_pos = []
    blank_pos = []
    for r in range(boardSize):
        for c in range(boardSize):
            if state.current_board[r][c] == white:
                white_num += 1
                if state.is_an_eye(r,c):
                    white_eyes += 1
                white_pos.append((r,c))

            elif state.current_board[r][c] == black:
                black_num += 1
                if state.is_an_eye(r,c):
                    black_eyes += 1 
                black_pos.append((r,c))
            else:
                blank_pos.append((r,c))
                if state.is_acquired_position(r,c,black):
                    black_acquired += 1
                if state.is_acquired_position(r,c,white):
                    white_acquired += 1
     #计算空白格属于什么棋子的liberty
    for blank_chess in blank_pos:
        blank_neighbors = state.find_neighbors(blank_chess[0],blank_chess[1])
        for neighbor in blank_neighbors:
            in_black_flag = False
            in_white_flag = False
            if neighbor in black_pos:
                black_liberty += 1
                in_black_flag = True
            if neighbor in white_pos:
                white_liberty += 1
                in_white_flag = True
            if in_black_flag or in_white_flag:
                break;
            
    # print("black liberty {} white liberty {} black_num {} white_num {}".format(black_liberty,white_liberty,black_num,white_num))
    black_score = black_num + black_liberty * 0.1 #+ black_acquired * 0.5 #- 0.8 * black_eyes
    white_score = white_num + white_liberty * 0.1 #+ white_acquired * 0.5 #- 0.8 * white_eyes

    diff = black_score - white_score
    if state.side == black:
        return diff
    return -1 * diff

       
class My_player:
    def __init__(self, board = None, side = None, alpha = 0.5, gamma = 0.8 ,initial_value = 0):
        self.board = board
        self.side = side
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self.history_states = []
        self.initial_value = initial_value

    def init_q_values(self):
        with open ('/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/train_set.json','r') as f:
            # lines = f.readlines()
            # for line in lines:
            try:
                data = json.load(f)
                self.q_values.update(data)
            except:
                print("no q values in json file")
        
    def Q(self, state):
        if state not in self.q_values:
            q = np.zeros((5,5))
            q.fill(self.initial_value)
            self.q_values[state] = q
        return self.q_values[state]


    def get_best_move(self):
        legal_moves = self.board.legal_moves()
        best_moves=[]
        
        #如果没有legal move就返回pass
        if not legal_moves:
            return -1, -1

        state = self.encode_board(self.board.side, self.board.current_board)
        q_values = self.Q(state)

        if np.count_nonzero(q_values) == 0:
            
            if(read_moves()<=2):
                point = random.choice(legal_moves)
                print("-------------this move is made by random----------------")
                return point[0], point[1]
            else:
                
                start = time.time()
                print("legal moves: {}".format(legal_moves))
                best_move = self.board.alpha_beta_search(self.board,2)
                end = time.time()
                print("time cost is {}".format(end-start))
                print("best move: {}".format(best_move))
                if best_move== None:
                    return -1, -1
                return best_move[0], best_move[1]
            
        #选择一个在legal move里面并且是最大的q值的坐标点
        while True:
            row, col  = self.find_max_coord(q_values)
            point = (row,col)
            if point in legal_moves:
                return point[0],point[1]
            #如果没有在legal move 就说明当前点不可以选，更新这个点的qvalue为-1，即在当前棋面下这个点永远不走
            else:
                q_values[point[0]][point[1]] = -1

    def find_max_coord(self,q):
        max_q = MIN
        row = 0
        col = 0
        for i in range(boardSize):
            for j in range(boardSize):
                if q[i][j] > max_q:
                    max_q = q[i][j]
                    row, col = i, j
        return row,col

    def make_one_move(self,row,col):
        if row == col == -1:
            self.board.set_passed_step(self.side)
            return "PASS"
        else:
            self.board.place_stone(row,col)
            #check if has dead_piece and update the board 
            self.board.update_game_state()
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
        with open ('/Users/xiaodongzheng/OneDrive - University of Southern California/USC/Classes/CSCI 561 Artificial Intelligence/HW/HW2/random_player_battle/history_states.txt','r') as file:
            lines = file.readlines()
            return lines

    def encode_board(self, side, board):
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

        state = ''.join([str(board_matrix[j][i])+" " for j in range(boardSize) for i in range(boardSize)])
        
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
        max_q = MIN
        for hist in self.history_states:
            l = hist.split("#")
            #get the board state information
            board_state = l[0]
            #get the step coordinate 落子坐标
            row = int(l[1])
            col = int(l[2])
            if row == -1 and col == -1:
                continue
            #获取状态奖励
            state_reward = self.get_state_reward(board_state)
            # print("state {} state reward {}\n".format(board_state,state_reward))
            
            #获取q value matrix
            q = self.Q(board_state)
            
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
        #check if the game is on the first step
        if(self.board.is_start_of_game()):
            write_moves("0")
            row, col = self.get_best_move()
            #make one move and record the state
            action = self.make_one_move(row, col)
            #update the current moves number of our side
            increase_move()
            print("current_moves: {}".format(self.board.num_moves))
            state = self.encode_board(side, self.board.current_board)
            self.save_states(state+"#"+ str(row)+ "#" + str(col))
        else:# the game is not the first step
            #if the same board then the op is passed
            if self.board.is_same_board():
                self.board.op_passed_move = True

            #read the history states played in former steps
            states = self.read_states()
            for state in states:
                self.history_states.append(state.rstrip('\n'))
            
            #read q values from file
            self.init_q_values()
            #get the best move
            action = self.get_best_move()
            #if the action is pass then pass and increase move count by 1
            if action[0] == -1 and action[1] == -1:
                self.board.set_passed_step(self.side)
                increase_move()
                #set the num_moves since pass also count
                self.board.num_moves = read_moves()
                print("current_moves: {} this move is PASS".format(self.board.num_moves))
                action = "PASS"
                #record passing step
                state = self.encode_board(side, self.board.current_board)
                self.save_states(state + "#"+ "-1" + "#" + "-1")
                self.history_states.append(state+"#" + "-1" + "#" + "-1")
                
                #check if game is ended
                if self.board.is_game_ended():
                    result = self.board.check_game_status()
                    self.train_after_end(result)
                    self.clean_up_after_end(result)
                    print("predicted result: {}".format(result))
            else:
                #make a move and increased the move count
                action = self.make_one_move(action[0],action[1])
                increase_move()
                print("total_moves: {} this move {}".format(self.board.num_moves,(action[0],action[1])))
                state = self.encode_board(side, self.board.current_board)#  +"#"+ str(action[0])+ "#" + str(action[1])
                #save state to file and append the state to history states
                self.save_states(state + "#"+ str(action[0])+ "#" + str(action[1]))
                self.history_states.append(state+"#"+ str(action[0])+ "#" + str(action[1]))
                
                #check if game is ended after the move,train if game is ended
                if self.board.is_game_ended():
                    result = self.board.check_game_status()
                    self.train_after_end(result)
                    self.clean_up_after_end(result)
                    print("predicted result: {}".format(result)) 

        return action
        
if __name__ == "__main__":

    #initialize the board 
    side, last_board, current_board = readInput()
    board = Board(boardSize)
    board.set_board(side,last_board, current_board)
    #create a player
    player = My_player(board,side)
    action = player.play()
    #output the move
    writeOutput(action)

 
    




