import sys, pygame, random, copy, time
import numpy as np
import scipy.stats as st

##########################################################
#INITIALIZE PYGAME
pygame.init()
size = width, height = 1000, 1000
screen = pygame.display.set_mode(size)
black = 0, 0, 0

MAX_COLOR = 150
white = MAX_COLOR, MAX_COLOR, MAX_COLOR
#screen.fill(black)
##########################################################

############## UTILS #####################################

def make_grid(grid_size):
    grid = np.ones((int(height/grid_size), int(width/grid_size)))

    return grid




def get_states(n, states = [""]):
    if n == 0:
        return states
    else:
        add_states = []
        for state in states:
            add_states.append(state + "0")
            add_states.append(state + "1")
        return get_states(n-1, add_states)

def make_rules(n):
    states = get_states(n)
    dct = {}
    for state in states:
        f = state[:len(state)-1]
        if (f not in dct):
            dct[f] = []
        dct[f].append(int(state[len(state)-1]))
    return dct



def basic_rules(n):
    states = get_states(n)
    dct = {}
    for state in states:
        dct[state] = 0
    return dct


###########################################################

class Cell:
    def __init__(self, x,y,size, rules):
        self.x=x
        self.y=y
        self.size=size

        self.rules = rules

        self.neighbors = []
        self.current_state=0

        cs = self.current_state * MAX_COLOR
        self.color = (cs,cs,cs)
        self.next_state = self.current_state


    def change_state(self, state):
        assert(state >= 0 and state <= 1 )

        self.current_state = state 
        self.color = (state*MAX_COLOR,state*MAX_COLOR,state*MAX_COLOR)


    def change_rules(self, rules):

        for r in rules:

            self.rules[r] = 1 - self.rules[r]



    def draw(self):

    

        pygame.draw.rect(screen,self.color,(self.x, 
                self.y, self.size,self.size ))


    def apply_rule(self):
        neighbors_states = "".join([str(n.current_state) for n in self.neighbors])
        #print(self.rules[neighbors_states])
        #print(neighbors_states, self.rules[neighbors_states] )
        self.next_state = self.rules[neighbors_states]



    def update(self):
        #self.apply_rule()

        self.change_state(self.next_state)

        self.draw()



class Grid():
    def __init__(self,grid_size,margin, rules):
        grid = make_grid(grid_size)
        self.cell_states = []

        self.cells = []
        self.margin = margin
        for row in range(int(height/grid_size)):
            cell_row = []

            for col in range(int(width/grid_size)):
                cell = Cell(col*grid_size, row*grid_size, grid_size, copy.deepcopy(rules))
                cell_row.append(cell)
            self.cells.append(cell_row)



        for row in range(margin,len(self.cells)-margin):
            state_row = []

            for col in range(margin,len(self.cells[0])-margin):
                state_row.append(str(cell.current_state))

                nbs = self.get_neighbors(row,col,margin)
                for n in nbs:
                    self.cells[row][col].neighbors.append(self.cells[n[0]][n[1]])
            self.cell_states.append("".join(state_row))


    def get_neighbors(self, row,col,margin):
        nbs = []
        for i in range(-margin,margin+1):
            for j in range(-margin,margin+1): 
                if not ((i==0) and (j==0)):
                    nbs.append((row+i,col+j))


        return nbs



    def change_rules(self, cell_row,cell_col,rules):
        assert(cell_row < len(self.cells) - self.margin)
        assert(cell_col < len(self.cells[0]) - self.margin)
        assert(cell_row  >= margin)
        assert(cell_col  >= margin)
        self.cells[cell_row][cell_col].change_rule(rules)


    def change_all_rules(self, rules):
        for row in range(self.margin,len(self.cells)-self.margin):
            for col in range(self.margin,len(self.cells[0])-self.margin):
                #self.cells[row][col].change_rules(rules)
                                #cell.change_rules(rules)

                cell = self.cells[row][col]
                cell.change_rules(rules)

                self.cells[row][col] = cell


    def change_state(self, cell_row,cell_col,state):
        assert(cell_row < len(self.cells) - self.margin)
        assert(cell_col < len(self.cells[0]) - self.margin)
        assert(cell_row  >= margin)
        assert(cell_col  >= margin)
        self.cells[cell_row][cell_col].change_state(state)


    def update(self):
        for row in range(margin,len(self.cells)-margin):
            for col in range(margin,len(self.cells[0])-margin):
                cell = self.cells[row][col]
                #print(cell.rules["00000000"])

                #cell.update()
                cell.apply_rule()


        self.cell_states = []
        for row in range(margin,len(self.cells)-margin):
            row_states = []
            for col in range(margin,len(self.cells[0])-margin):
                cell = self.cells[row][col]
                cell.update()
                row_states.append(str(cell.current_state))
            self.cell_states.append("".join(row_states))


        pygame.display.update()




def main(grid_size, margin):

    all_rules = basic_rules(8)
    all_rule_keys = sorted(all_rules.keys(), key=lambda k: int(k,2))
    simple = all_rule_keys[0:1]+ ["10101010"] + ["01010101"] + ["11111111"]
    simple += ["01101101"] + ["10010010"] + ["11110000"] + ["00100100"]
    simple += ["00000001"] + ["10000000"] + ["01111111"] + ["11111110"]

    #print(all_rules)

    grid = Grid(grid_size,margin, all_rules)

    history = []
    for i in range(-len(grid.cells)/4,len(grid.cells)/4+1):
        #grid.change_state(int(len(grid.cells)/2)+i+1,
           # int(len(grid.cells[0])/2),1)
        #grid.change_state(int(len(grid.cells)/2),
             #   int(len(grid.cells[0])/2)+i+1,1)
        for j in range(-len(grid.cells[0])/4,len(grid.cells[0])/4+1):
            grid.change_state(int(len(grid.cells)/2)+
                j,int(len(grid.cells[0])/2)+j,1)
            grid.change_state(int(len(grid.cells)/2),
                int(len(grid.cells[0])/2)+j,1)
            grid.change_state(int(len(grid.cells)/2)+j,
                int(len(grid.cells[0])/2),1)
           # grid.change_state(int(len(grid.cells)/2)+i,
             #   int(len(grid.cells[0])/2)+i,1)

            grid.change_state(int(len(grid.cells)/2)-i,
                int(len(grid.cells[0])/2)+i,1)

    #grid.change_all_rules(["00000000"])
    #grid.change_all_rules(["11111111"])
    #grid.change_all_rules(["00100100"])
    #grid.change_all_rules(["11011011"])

    #grid.change_all_rules(["01010101"])
    #grid.change_all_rules(["10101010"])
    #grid.change_all_rules(["00110011"])
    #grid.change_all_rules(["01001001"])
    #grid.change_all_rules(["11001100"])
    #grid.change_all_rules(["10010010"])
    #grid.change_all_rules(["10010100"])

    grid.change_all_rules(["11111111"])

    #grid.change_all_rules(["11111000"])
    #grid.change_all_rules(["00110011"])
    #grid.change_all_rules(["10100101"])

    #grid.change_all_rules(["11011011"])
    #grid.change_all_rules(["01111111"])

   # grid.change_all_rules(["11111110"])
    grid.change_all_rules(["00000001"])
    #grid.change_all_rules(["11000011"])

    grid.change_all_rules(["01111110"])
    #grid.change_all_rules(["00100000"])
    #grid.change_all_rules(["00000100"])
    grid.change_all_rules(["10000000"])
    #grid.change_all_rules(["11111110"])
    #grid.change_all_rules(["01111111"])
   # grid.change_all_rules(["11011111"])
   # grid.change_all_rules(["11111011"])

    #grid.change_all_rules(["01010101"])

   # grid.change_all_rules(["01111111"])

    favorite_starts = [["11111111", "00000001", "01111110", "10000000"]]

    grid.update()


    t_last = time.time()
    t_delay = 0.1
    pause, drawing,count = True, False,0

    while True:
        if time.time() - t_last > t_delay and not pause:
            grid.update()
            t_last = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:

                    grid.update()
                elif event.key == pygame.K_RETURN:
                    change = [simple[random.randint(0,len(simple)-1)]]
                    #change = ["00000000"]
                    print ("NEW: ", change)

                    #change = [all_rule_keys[random.randint(0,len(all_rule_keys)-1)]]                    dct = {}
                    dct = {}
                    for row in range(margin,len(grid.cells)-margin):
                        for col in range(margin,len(grid.cells[0])-margin):
                            dct[(row,col)] = change
                    history.append(copy.deepcopy(dct))
                    grid.change_all_rules(change)

                elif event.key == pygame.K_c:
                    z = 0
                    while z < 500:
                        change = [all_rule_keys[random.randint(0,len(all_rule_keys)-1)]]

                        dct = {}
                        for row in range(margin,len(grid.cells)-margin):
                            for col in range(margin,len(grid.cells[0])-margin):
                                dct[(row,col)] = change
                        grid.change_all_rules(change)


                        same = False

                        curr_state = grid.cell_states
                        for k in range(10):
                            grid.update()

                            new_state = grid.cell_states
                            if new_state == curr_state:
                                same= True

                        if (not same):
                            print(len(curr_state))
                            print("*"*100)
                            print(len(new_state))
                            print("New: ", change)
                            break
                        else:
                            grid.change_all_rules(change)

                            history.append(copy.deepcopy(dct))

                        if z % 20 == 0:
                            print(z)
                        z += 1


                elif event.key == pygame.K_d:
                    print("d")
                    z = 0
                    while z < 100:
                        change = [all_rule_keys[random.randint(0,len(all_rule_keys)-1)]]

                        dct = {}
                        for row in range(margin,len(grid.cells)-margin):
                            for col in range(margin,len(grid.cells[0])-margin):
                                dct[(row,col)] = change
                        grid.change_all_rules(change)

                        curr_state = grid.cell_states
                        grid.update()
                        new_state = grid.cell_states
                        if (curr_state == new_state):
                            print(len(curr_state))
                            print("*"*100)
                            print(len(new_state))
                            print("New: ", change)
                            history.append(copy.deepcopy(dct))

                            break
                        else:
                            grid.change_all_rules(change)


                        if z % 20 == 0:
                            print(z)
                        z += 1

                elif event.key == pygame.K_p:
                    if pause:
                        pause = False
                    else:
                        pause = True

                elif event.key == pygame.K_k:
                    grid.change_all_rules(["00000000"])
                    #grid.change_all_rules(["11111111"])

                elif event.key == pygame.K_f:
                    t_delay = t_delay / 2.
                elif event.key == pygame.K_s:
                    t_delay = t_delay * 2.

                elif event.key == pygame.K_r:
                    grid.change_state(margin,margin,1)
                    grid.change_state(margin+1,margin,0)



if __name__ == "__main__":
    grid_size = 5
    margin=1
    main(grid_size, margin)
