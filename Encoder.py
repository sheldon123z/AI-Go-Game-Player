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
              if board[r][c] == side:
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


# def alpha_beta_select(board,legal_moves,max_depth,max_breadth):
#     best_moves =[]
#     max_score = None
#     black_best = MIN
#     white_best = MIN
#     num = max_breadth
#     possible_moves = set()
#     n = max_breadth
    
#     if len(legal_moves) < n:
#         possible_moves.update(legal_moves)
#     else:
#         while len(possible_moves) <n:
#             possible_moves.add(random.choice(legal_moves))
    
#     print("possible moves {}".format(possible_moves))
#     for move in possible_moves:
#         next_state = board.make_test_move(3-board.side,move[0],move[1])
#         op_best_result = alpha_beta_result(next_state,max_depth,black_best,white_best,evaluation,possible_moves)
#         # print("op best result:{} for move: {}".format(op_best_result,move))
#         my_best_result = -1 * op_best_result
#         print("my best result:{} for move: {}".format(my_best_result,move))
#         if not best_moves or my_best_result > max_score:
#             best_moves.append(move)
#             max_score = my_best_result
#             if board.side == white:
#                 black_best = max_score
#             elif board.side == black:
#                 white_best = max_score
#         elif my_best_result == max_score:
#             best_moves.append(move)
#     return best_moves

# def alpha_beta_result(state,max_depth,best_black,best_white,eva_function,possible_moves):
    
#     if read_moves() + max_depth == state.max_moves or (state.op_passed_move and state.me_passed_move):
#         result = state.check_game_status()
#         if result == LOSE:
#             return MIN
#         elif result == WIN:
#             return MAX
#     if max_depth == 0 or len(possible_moves) == 0:
#         result = eva_function(state)
#         return result
    
#     best_score_so_far = MIN
#     count = 0

#     avalible_moves = possible_moves

#     for move in avalible_moves:
#         next_state = state.make_test_move(3-state.side,move[0],move[1])
#         op_best = alpha_beta_result(next_state,max_depth - 1,best_black,best_white,eva_function,avalible_moves)
#         # print("op_best:{}".format(op_best), type(op_best))
#         my_best = -1 * op_best
#         if my_best > best_score_so_far:
#             best_score_so_far = my_best

#             #the next player is black
#         if state.side == white:
#             #record current best black for next round
#             if best_score_so_far > best_black:
#                 best_black = best_score_so_far
#             #if the current white score less than best white then break 
#             current_white_score = -1 * best_score_so_far
#             if current_white_score < best_white:
#                 break
#             #if next player is white then choose the best result for black
#         elif state.side == black:
#             #record the current best white for next round
#             if best_score_so_far > best_white:
#                 best_white = best_score_so_far
#             current_black = -1 * best_score_so_far
#             #if the current player's best score less than upper, break and return
#             if current_black < best_black:
#                 break

#     return best_score_so_far