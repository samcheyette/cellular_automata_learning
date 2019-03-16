import sys, pygame, random, copy, time
import numpy as np
import scipy.stats as st

##########################################################
#INITIALIZE PYGAME
pygame.init()
size = width, height = 1000, 1000
screen = pygame.display.set_mode(size)
black = 0, 0, 0

MAX_COLOR = 255
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

def basic_rules(n):
    states = get_states(n)
    dct = {}
    for state in states:
        dct[state] = np.random.normal(0,1.)
    return dct

def inputs_targets_1(n_cell, n_ex):
    inputs, targets = [],[]

    for i in range(n_ex):
        inp = [0 for _ in range(n_cell)]
        tgt = [0 for _ in range(n_cell)]
        flip = random.randint(0,n_cell-1)
        inp[flip] = 1
        for j in range(flip+1):
            tgt[j] = 1

        inputs.append(inp)
        targets.append(tgt)

    return inputs,targets




def inputs_targets_2(n_cell, n_ex, offset=2):
    inputs, targets = [],[]

    for i in range(n_ex):
        #in_0 = "0000000000000" +
        in_0 =  "0" * offset + "{0:b}".format(i) 
        inp = []
        for j in in_0:
            inp.append(int(j))

        #in_0 = [int(j) for j in in_0]
        inp += [0 for _ in range(n_cell-len(inp))]
        tgt = [0 for _ in range(n_cell)]

        for j in range(i):
            tgt[j] = 1

        inputs.append(inp)
        targets.append(tgt)

    return inputs,targets



def inputs_targets_3(n_cell, n_ex):
    inputs, targets = [],[]

    for i in range(n_ex):
        inp = [0 for _ in range(n_cell)]
        tgt = [0 for _ in range(n_cell)]
        flip = random.randint(0,int(n_cell/2)-2)
        plus = random.randint(flip+1,int(n_cell/2)-1)
        inp[flip] = 1
        inp[plus] = 1
        for j in range(flip+plus+1):
            tgt[j] = 1

        inputs.append(inp)
        targets.append(tgt)

    return inputs,targets



def inputs_targets_4(n_cell, n_ex):
    inputs, targets = [],[]

    for i in range(n_ex):
        inp = [0 for _ in range(n_cell)]
        tgt = [0 for _ in range(n_cell)]
        flip = random.randint(0,int(n_cell)-1)
        inp[flip] = 1
        for j in range(int(n_cell)):
            tgt[j] = ((j+flip) % 2)

        inputs.append(inp)
        targets.append(tgt)

    return inputs,targets

##########################CELL###############################


class Cell:
    def __init__(self, x,y,size, rules):
        self.x=x
        self.y=y
        self.size=size

        self.rules = rules

        self.neighbors = []
        self.current_state=0.0

        cs = int(self.current_state * MAX_COLOR)
        self.color = (cs,cs,cs)
        self.next_state = self.current_state


    def change_state(self, state):
        assert(state >= 0 and state <= 1 )

        self.current_state = state 
        sm = int(state*MAX_COLOR)
        self.color = (sm,sm,sm)


    """


    def change_rules(self, rules, mult):


        for r in rules:
            if self.rules[r] > 1.:
                self.rules[r] = 1.
            if self.rules[r] < 0.01:
                self.rules[r] = 0.01
            self.rules[r] = self.rules[r] * mult
    """

    def change_rules(self, rules, add):


        for r in rules:
            self.rules[r] = self.rules[r] + add


    def draw(self):

    

        pygame.draw.rect(screen,self.color,(self.x, 
                self.y, self.size,self.size ))


    def apply_rule(self):
        neighbors_states = [n.current_state for n in self.neighbors]
        #print(self.rules[neighbors_states])
        #print(neighbors_states, self.rules[neighbors_states] )
        self.next_state = 0.


        for r in self.rules:
            #p_r = min(1.,max(0., self.rules[r]))
            p_r = min(1.,max(0.,(1/(1+np.exp(-self.rules[r])))))
            p = 1.
            for s in range(len(r)):
                f_s = float(r[s])
                p1 = f_s * neighbors_states[s]
                p2 = (1. - f_s) * (1.-neighbors_states[s])

                p_s = p1 + p2 
                p *= p_s

                
            #print (p_r)
            self.next_state += p * p_r
                #p_s = float(s)



        if self.next_state < 0 or self.next_state > 1:
            print "BAD?", self.next_state, neighbors_states, self.rules
            print 

        self.next_state = min(1.,max(0.,self.next_state))
            #assert(False)



    def update(self, draw=True):
        self.apply_rule()

        self.change_state(self.next_state)
        if draw:
            self.draw()





##########################GRID###############################


class Grid():
    def __init__(self,grid_size,rules):
        grid = make_grid(grid_size)
        self.cell_states = []

        self.cells = []
        self.input_cells = []
        self.output_cells = []
        self.n_row = 0
        self.n_col =int(height/grid_size)

        for row in range(int(height/grid_size)):
            cell_row = []

            for col in range(int(width/grid_size)):
                cell = Cell(col*grid_size, row*grid_size, grid_size, copy.deepcopy(rules))
                cell_row.append(cell)
            self.cells.append(cell_row)

        for row in range(0,len(self.cells)):
            state_row = []

            for col in range(0,len(self.cells[0])):
                state_row.append(cell.current_state)
                if row < len(self.cells) -1:
                    nbs = self.get_neighbors(row,col)
                    for n in nbs:
                        self.cells[row][col].neighbors.append(self.cells[n[0]][n[1]])
                else:
                    self.input_cells.append(self.cells[row][col])
            self.cell_states.append(copy.copy(state_row))

    def change_state(self, cell_row,cell_col,state):

        self.cells[cell_row][cell_col].change_state(state)
        self.cell_states[cell_row][cell_col] = state
        
    def get_neighbors(self, row,col):
        nbs = []
        n_row,n_col = len(self.cells), len(self.cells[0])
        for j in range(-1,2):
            if ((col+j < 0)):
                nbs.append((row+1,0))

            elif ((col+j >= n_col)):
                nbs.append((row+1,n_col-1))

            else:
                nbs.append((row+1,col+j))

        return nbs

    def add_inputs(self, states):
        for c in range(len(states)):
            cell = self.input_cells[c]
            cell.change_state(states[c])
            cell.draw()

    def reset(self,draw=True):
        for row in range(0,len(self.cells)):
            for col in range(0,len(self.cells[0])):

                cell = self.cells[row][col]
                self.change_state(row, col,.5)
                #self.cell_states[row][col] = 0

                if draw:
                    cell.draw()
        if draw:
            pygame.display.update()

    def change_all_rules(self, rules, mult):
        for row in range(len(self.cells)-1):
            for col in range(0,len(self.cells[0])):

                cell = self.cells[row][col]
                cell.change_rules(rules, mult)


                

    def change_row_rules(self, row, rules, mult):
        for col in range(0,len(self.cells[0])):


            cell = self.cells[row][col]
            cell.change_rules(rules, mult)



    def change_col_rules(self, col, rules,mult):
        for row in range(len(self.cells)-1):


            cell = self.cells[row][col]
            cell.change_rules(rules, mult)

    def change_cell_rules(self, row,col, rules, mult):
        cell = self.cells[row][col]
        cell.change_rules(rules,mult)

    def draw_targets(self, targets):
        for i in range(len(self.cells[0])):
            cell = self.cells[0][i]
            color = (int(targets[i])*MAX_COLOR,0,0)
            pygame.draw.rect(screen, color,(cell.x,cell.y,cell.size,cell.size+1),
                    3)


    def update(self, delay=0.1, draw=True):
        states = ""
        for row in range(len(self.cells)-2,-1,-1):
            for col in range(0,len(self.cells[0])):
                cell = self.cells[row][col]
                cell.apply_rule()
                cell.update()
                self.cell_states[row][col] = cell.current_state


            if draw:
                pygame.display.update()

                t = time.time()
                while time.time()  - t  < delay:
                    pass



def run_grid(grid, inputs, targets, delay, draw):

    acc = 0.
    for t in range(len(targets)):
        grid.reset()
        grid.add_inputs(inputs[t])
        target = targets[t]
        if draw:
            grid.draw_targets(target)
        grid.update(delay=delay, draw=draw)
        if draw:
            grid.draw_targets(target)
            tm = time.time()
            while time.time() - tm < 0.2:
                pass

        output = grid.cell_states[0]

        #p = st.bernoulli.logpmf(target,output)
        p = (target - np.array(output))**2. 

        acc += -np.log2(np.sum(p)+1.)
       # acc += np.sum(p)

    acc = 2.**(acc)
    return(acc)

def mh_step(grid, curr_rules,curr_post, all_rules, inputs,targets, delay,draw,momentum):
    grid.reset()
    pygame.display.update()
   # pr = curr_rules[random.randint(0,len(curr_rules)-1)]
    proposal_rule= [all_rules[random.randint(0,len(all_rules)-1)]]

    proposal_row, proposal_col = -1,-1 #need to implement


    if momentum != None and random.random() < 0.99:
        mult = momentum[3]
        proposal_rule = momentum[2]
        proposal_row= momentum[0]
        proposal_col = momentum[1]

    #print(proposal_row,proposal_col)
    else:
        r_row, r_col = random.random() < 0.25, random.random() < 0.1

        #mult = max(0.25,min(5.,np.random.normal(1.,0.25)))
        mult = np.random.normal(0.,1.)

        if r_row:
            if random.random() < 0.1:
                proposal_row = random.randint(0,int(len(grid.cells)/2))
            elif random.random() < 0.1:
                proposal_row = len(grid.cells) - 1

            else:
                proposal_row = random.randint(0,len(grid.cells)-2)
        if r_col:
            proposal_col = random.randint(0, len(grid.cells[0])-1)


    if proposal_row != -1 and proposal_col != -1:

        grid.change_cell_rules(proposal_row,proposal_col,proposal_rule,mult)
    elif proposal_row != -1 and proposal_col == -1:
        grid.change_row_rules(proposal_row, proposal_rule,mult)
    elif proposal_row == -1 and proposal_col != -1:
        grid.change_col_rules(proposal_col, proposal_rule,mult)
    else:
        grid.change_all_rules(proposal_rule,mult)

    acc = run_grid(grid, inputs, targets, delay=0.001,draw=False)

    if curr_post < -10e9:
        ratio = 1
        rand = 0
    else:
        ratio = acc/curr_post 
        rand = random.random()
        #rand = 1.

    if ratio < rand*1.1:
    #if acc < curr_post:
        momentum = None
        if proposal_row != -1 and proposal_col != -1:
            grid.change_cell_rules(proposal_row, proposal_col,proposal_rule, -mult)
        elif proposal_row == -1 and proposal_col != -1:
            grid.change_col_rules(proposal_col, proposal_rule, -mult)
        elif proposal_row != -1 and proposal_col == -1:
            grid.change_row_rules(proposal_row, proposal_rule, -mult)
        else:
            grid.change_all_rules(proposal_rule, -mult)

    else: 
        if acc > curr_post:         
            momentum = (proposal_row, proposal_col, proposal_rule, mult*1.1)
        else:
            momentum = None

        curr_rules.append((proposal_row, proposal_col, proposal_rule, mult))

        curr_post = acc

        #print (curr_rules)
        print curr_post, momentum
    return curr_rules, curr_post, momentum






def main(n_cells, grid_size):

    all_rules = basic_rules(3)
    start_rules = copy.deepcopy(all_rules)
    all_rule_keys = all_rules.keys()

    grid = Grid(grid_size, start_rules)

    #inputs = np.random.random((5,n_cells))
    #inputs = np.random.randint(0,2,(5,n_cells))

    inputs,targets = inputs_targets_4(n_cells, 8)
    validation_inputs,validation_targets = inputs_targets_4(n_cells, 8)

    print(inputs)
    print(targets)
    grid.change_all_rules(["010"], 5.)
    #grid.change_all_rules(["011"], 5.)
    #grid.change_all_rules(["110"], 5.)

    #grid.update()

    curr_rules = []
    curr_post = -float("inf")
    draw,delay = False, 0.01
    pause = True
    momentum = None




    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    acc = 0.
                    acc = run_grid(grid, inputs, targets,0.1,True)
                        #acc += np.sum(p)
                    print "Target accuracy: ", acc
                    print

                if event.key == pygame.K_v:
                    acc=0.
                    acc = run_grid(grid, validation_inputs, validation_targets,0.1,True)

                    print "Validation accuracy: ", acc
                    print



                elif event.key == pygame.K_p:
                    if pause:
                        pause = False
                    else:
                        pause = True


                #pygame.display.update()

               # grid.update(0.1)
        if not pause:
            curr_rules, curr_post,momentum = mh_step(grid,curr_rules,curr_post, 
                all_rule_keys, inputs,targets, delay,draw,momentum)
            #print(curr_post)
            #if curr_post > 0.99:
              #  pause = True

if __name__ == "__main__":
    n_cells = 12
    grid_size = int(width/n_cells)
    main(n_cells, grid_size)

