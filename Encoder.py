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
