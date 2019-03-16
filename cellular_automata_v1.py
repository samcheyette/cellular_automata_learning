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

def levenshtein(seq1, seq2):
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]


def hamming(s1,s2):
    assert(len(s1) == len(s2))
    c = 0
    for i in range(len(s1)):
        if (s1[i] != s2[i]):
            c += 1
    return c

def equi_dist(s1,s2):
    assert(len(s1) == len(s2))
    n0,n1 = 1.,1.
    c0,c1 = 1.,1.
    n_t,n_c = 0.,0.


    for i in range(len(s1)):
        n_t += 1
        if (s1[i] == s2[i]):
            if s1[i] == 0:
                c0 += 1
            else:
                c1 += 1
            n_c += 1.
        if (s2[i] == 0):
            n0 += 1
        else:
            n1 += 1


    p0 = ((n0-c0)/n0) * n_t * 0.5
    p1 = ((n1-c1)/n1) * n_t * 0.5
    pc = (n_t - n_c)

    p = 0.5 * (p0 + p1) + 0.5 * pc


    return p

def inputs_targets_1(n_cell, n_ex):
    inputs, targets = [],[]

    for i in range(n_ex):
        inp = [str(0) for _ in range(n_cell-2)]
        tgt = [str(0) for _ in range(n_cell-2)]
        flip = random.randint(0,n_cell-14)
        for j in range(9):
            inp[flip+j] = "1"
            tgt[flip+j+3] = "1"


        inp = "0" + "".join(inp) + "0"
        tgt = "0" + "".join(tgt) + "0"
        inputs.append(inp)
        targets.append(tgt)

    return inputs,targets



def inputs_targets_2(n_cell, n_ex):
    inputs, targets = [],[]

    for i in range(n_ex):
        inp = [str(0) for _ in range(n_cell-2)]
        tgt = [str(1) for _ in range(n_cell-2)]
        flip = random.randint(0,n_cell-11)
        for j in range(9):
            inp[flip+j] = "1"
            tgt[flip+j] = "0"


        inp = "0" + "".join(inp) + "0"
        tgt = "0" + "".join(tgt) + "0"
        inputs.append(inp)
        targets.append(tgt)

    return inputs,targets


def inputs_targets_3(n_cell, n_ex):
    inputs, targets = [],[]

    for i in range(n_ex):
        inp = [str(0) for _ in range(n_cell-2)]
        tgt = [str(0) for _ in range(n_cell-2)]
        flip = random.randint(0,n_cell-3)
        inp[flip] = "1"
        for j in range(flip+1):
            tgt[j] = "1"

        inp = "0" + "".join(inp) + "0"
        tgt = "0" + "".join(tgt) + "0"
        inputs.append(inp)
        targets.append(tgt)

    return inputs,targets


def inputs_targets_4(n_cell, n_ex):
    inputs, targets = [],[]

    for i in range(n_ex):
        #in_0 = "0000000000000" +
        in_0 = ""
        in_0 +=  "{0:b}".format(i) 
        inp =  [in_0] + [str(0) for _ in range(n_cell-2-len(in_0))]
        tgt = [str(0) for _ in range(n_cell-2)]

        for j in range(i):
            tgt[j] = "1"

        inp = "0" + "".join(inp) + "0"
        tgt = "0" + "".join(tgt) + "0"
        inputs.append(inp)
        targets.append(tgt)

    return inputs,targets




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
        self.input_cells = []
        self.output_cells = []
        self.margin = margin
        for row in range(int(height/grid_size)):
            cell_row = []

            for col in range(int(width/grid_size)):
                cell = Cell(col*grid_size, row*grid_size, grid_size, copy.deepcopy(rules))
                cell_row.append(cell)
            self.cells.append(cell_row)



        for row in range(margin-1,len(self.cells)-margin):
            state_row = []

            for col in range(margin,len(self.cells[0])-margin):
                state_row.append(str(cell.current_state))

                nbs = self.get_neighbors(row,col,margin)
                for n in nbs:
                    self.cells[row][col].neighbors.append(self.cells[n[0]][n[1]])
            self.cell_states.append("".join(state_row))

        

        for col in range(len(self.cells[0])):
            self.input_cells.append(self.cells[len(self.cells)-margin][col])
        
        self.current_row = len(self.cells)-margin-1


    def get_neighbors(self, row,col,margin):
        nbs = []
        for j in range(-1,2):
            nbs.append((row+1,col+j))


        #for i in range(-margin,margin+1):
            ##for j in range(-margin,margin+1): 
                #if not ((i==0) and (j==0)):
                    #nbs.append((row+i,col+j))


        return nbs



    def change_rules(self, cell_row,cell_col,rules):
        assert(cell_row < len(self.cells) - self.margin)
        assert(cell_col < len(self.cells[0]) - self.margin)
        assert(cell_row  >= margin-1)
        assert(cell_col  >= margin)
        self.cells[cell_row][cell_col].change_rule(rules)




    def reset(self):
        for row in range(self.margin-1,len(self.cells)-self.margin):
            for col in range(self.margin,len(self.cells[0])-self.margin):
                cell = self.cells[row][col]
                self.change_state(row, col,0)
                cell.draw()

        pygame.display.update()


    def reset_rules(self, rules):
        for row in range(self.margin-1,len(self.cells)-self.margin):
            for col in range(self.margin,len(self.cells[0])-self.margin):
                cell = self.cells[row][col]
                cell.rules = copy.deepcopy(rules)

    def change_all_rules(self, rules):
        for row in range(self.margin-1,len(self.cells)-self.margin):
            for col in range(self.margin,len(self.cells[0])-self.margin):

                cell = self.cells[row][col]
                cell.change_rules(rules)


                

    def change_row_rules(self, row, rules):
        for col in range(self.margin-1,len(self.cells[0])-self.margin):
                #self.cells[row][col].change_rules(rules)
                                #cell.change_rules(rules)

            cell = self.cells[row][col]
            cell.change_rules(rules)

            #self.cells[row][col] = cell
                

    def change_col_rules(self, col, rules):
        for row in range(self.margin,len(self.cells)-self.margin):
                #self.cells[row][col].change_rules(rules)
                                #cell.change_rules(rules)

            cell = self.cells[row][col]
            cell.change_rules(rules)

    def change_cell_rules(self, row,col, rules):
                #self.cells[row][col].change_rules(rules)
                                #cell.change_rules(rules)

        cell = self.cells[row][col]
        cell.change_rules(rules)

        #self.cells[row][col] = cell


    def draw_targets(self, targets):
        for i in range(len(self.cells[0])):
            cell = self.cells[0][i]
            color = (int(targets[i])*200,0,0)
            pygame.draw.rect(screen, color,(cell.x,cell.y,cell.size,cell.size+1),
                    2)


    def change_state(self, cell_row,cell_col,state):
        assert(cell_row < len(self.cells) - self.margin)
        assert(cell_col < len(self.cells[0]) - self.margin)
        assert(cell_row  >= margin-1)
        assert(cell_col  >= margin)
        self.cells[cell_row][cell_col].change_state(state)
        c_row = self.cell_states[cell_row]
        c_row = c_row[:cell_col-margin] + str(state) + c_row[cell_col-margin+1:]

        self.cell_states[cell_row] = c_row

    def add_input(self, states):
        for c in range(len(states)):
            cell = self.input_cells[c]
            cell.change_state(int(states[c]))
            cell.draw()

        pygame.display.update()

    """
    def update_sequentially(self):
        if self.current_row > margin:
            row = self.current_row
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
    """


    def update(self, delay=0.1):
        self.rendering = True
        for row in range(len(self.cells)-margin-1,margin-2,-1):
            for col in range(margin,len(self.cells[0])-margin):
                cell = self.cells[row][col]
                #print(cell.rules["00000000"])

                cell.apply_rule()
                cell.update()
            t = time.time()
            #pygame.time.delay(int(delay*1000))
            while time.time()  - t  < delay:
                pass
            pygame.display.update()


        self.cell_states = []
        for row in range(margin-1,len(self.cells)-margin):
            row_states = []
            for col in range(margin,len(self.cells[0])-margin):
                cell = self.cells[row][col]
                #cell.update()
                row_states.append(str(cell.current_state))
            self.cell_states.append("".join(row_states))


        #pygame.display.update()



def main(n_cells, grid_size, margin):

    #all_rules = basic_rules(8)
    all_rules = basic_rules(3)
    start_rules = copy.deepcopy(all_rules)
    print(all_rules)
    all_rule_keys = sorted(all_rules.keys(), key=lambda k: int(k,2))

    grid = Grid(grid_size,margin, all_rules)

    init_rules = []

    #init_rules = ["001", "011", "110", "111"]
    init_rules = ["100", "000", "001"]

    grid.change_all_rules(init_rules)


    #grid.change_state(8,5,1)
    #grid.change_state(15,5,1)
    #grid.change_state(15,4,1)
    #grid.change_state(15,10,1)
    #grid.change_state(15,11,1)
    #grid.add_input("00110"*2)
    print len(grid.cells[0])

    #inputs = ["01010101", "10101010","00000000","11111111"]
    #inputs = ["001001"*3, "000001"*3, "1"*18,"0"*18,"01"*9]
    #inputs = ["0" * (n_cells-2*margin), "1"*(n_cells-2*margin)]
    #inputs = ["0"+i+"0" for i in inputs]
    #targets = inputs
    #targets = ["0"*20,"0"*20,"0"*20,"0" + "001"*6 + "0","0"*20]
    inputs,targets = inputs_targets_3(n_cells,10)
    print(inputs)
    print(targets)
    distance = 0

    curr_posterior = None
    curr_distance = None
    curr_likelihood = None
    curr_rules = [(-1,-1,i) for i in init_rules]

    best_posterior = curr_posterior
    best_likelihood = curr_likelihood
    best_distance = curr_distance
    best_rules = [(-1,-1,i) for i in init_rules]

    print("D: ", distance)

    delay = 0.0001
    count = 0


    #grid.change_state(0,5,1)


    #t = time.time()
    #while time.time() - t < 5:
       # pass
    #sys.exit()


    while True:

        #sys.exit()
        #if not(grid.rendering):
        #grid.update(delay)


        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    grid.reset()
                if event.key == pygame.K_RIGHT:
                   # n = input("How many steps: ")
                    #print(n)
                    n = 100

                    for i in range(int(n)):
                        grid.reset()
                        pygame.display.update()


                        lcr = (1.-(1./(len(curr_rules) + 1.)))*0.4
                        print lcr
                        remove = (random.random() < lcr)
                        if ( remove):
                            pr = curr_rules[random.randint(0,len(curr_rules)-1)]
                            proposal_row,proposal_col,proposal_rule= pr
                            proposal_rule = [proposal_rule]
                            if proposal_row == -1:
                                grid.change_all_rules(proposal_rule)
                            else:
                                if proposal_col == -1:
                                    grid.change_row_rules(proposal_row,proposal_rule)
                                else:
                                    grid.change_cell_rules(proposal_row,proposal_col,proposal_rule)


                        else:

                            proposal_rule= [all_rule_keys[random.randint(0,len(all_rule_keys)-1)]]
                        
                            r_row = random.random() < 0.5
                            r_col = random.random() < 0.5
                            proposal_row, proposal_col = -1,-1 #need to implement


                            if r_row:
                                if random.random() < 0.25:
                                    proposal_row = 0
                                else:
                                    proposal_row = random.randint(margin-1,len(grid.cells)-margin-1)
                                if r_col:
                                    proposal_col = random.randint(margin, len(grid.cells[0])-margin-1)
                                    grid.change_cell_rules(proposal_row,proposal_col,proposal_rule)
                                else:
                                    grid.change_row_rules(proposal_row, proposal_rule)

                            elif r_col:
                                proposal_col = random.randint(margin, len(grid.cells[0])-margin-1)
                                grid.change_col_rules(proposal_col, proposal_rule)

                            else:
                                grid.change_all_rules(proposal_rule)

                            if (proposal_row,proposal_col,proposal_rule[0]) in curr_rules:
                                remove = True

                        print "PROPOSE: ", proposal_row,proposal_rule
                        if remove:
                            print "(REMOVAL)"

                        prop_distance = 1e-2
                        for j in range(len(inputs)):
                            grid.reset()
                            inp = inputs[j]
                            target = targets[j]

                            grid.draw_targets(target)
                            grid.add_input(inp)
                            grid.draw_targets(target)

                            grid.update(delay)
                           # pygame.display.update()

                            output ="0" +grid.cell_states[0] + "0"
                            prop_distance += equi_dist(output,target)**2.
                            print("I: ", inp)
                            print("O: ", output)
                            print("T: ", target)
                            print("")

                        C = (2*remove) - 1 - 0.001
                        prop_prior = -np.log2(len(curr_rules) - C + 1)
       
                        prop_likelihood =  -2*np.log2(prop_distance)
                        #prop_posterior = np.exp(prop_prior + prop_likelihood)
                        prop_posterior = (2.**(prop_prior)) * (2.**(prop_likelihood))

                        #output =grid.cell_states[margin-1]
                       # print(grid.cell_states)
                        #print(output,target)
                        print("D (prop): ", prop_distance)
                        print("D (curr): ", curr_distance)
                        print("Prior, Likelihood (Prop): ",2.**prop_prior, 2.**prop_likelihood )
                        print("CR: ", curr_rules)
                        print("*"*5)

                        if curr_posterior == None:
                            accept = True
                        else:
                            acc_ratio = prop_posterior / curr_posterior
                            accept = random.random() < acc_ratio

                        print(prop_posterior,curr_posterior,accept)
                        if accept:
                            curr_posterior = prop_posterior
                            curr_likelihood = prop_likelihood
                            curr_distance = prop_distance
                            pr = (proposal_row, proposal_col, proposal_rule[0])
                            if (pr in curr_rules) and remove:
                                print("REMOVED!!!!!!*!*!*!*!")
                                curr_rules.remove(pr)
                            else:
                                curr_rules.append(pr)

                            if curr_posterior > best_posterior:
                                best_likelihood = curr_likelihood
                                best_posterior = curr_posterior
                                best_distance = curr_distance
                                best_rules = copy.deepcopy(curr_rules)

                        else:
                            if proposal_row == -1 and proposal_col == -1:
                                grid.change_all_rules(proposal_rule)
                            elif proposal_row == -1:
                                grid.change_col_rules(proposal_col, proposal_rule)
                            else:
                                if proposal_col != -1:
                                    grid.change_cell_rules(proposal_row,proposal_col,proposal_rule)
                                else:
                                    grid.change_row_rules(proposal_row, proposal_rule)
                        
                        if (curr_distance < 2):
                            break


                    print("")

                    print("CURR:", curr_distance, curr_posterior)
                    print("BEST:", best_distance, best_posterior)
                    print("")
                    print("*" * 20)

                elif event.key == pygame.K_o:

                    grid.reset()
                    pygame.display.update()

                    grid.update(0.1)


                elif event.key == pygame.K_v:

                    #grid = Grid(grid_size,margin, copy.deepcopy(start_rules))
                    grid.reset()
                    grid.reset_rules(copy.deepcopy(start_rules))
                    #grid.change_all_rules(init_rules)
                    #grid.__init__(grid_size,margin,copy.deepcopy(start_rules))
                    #for rule in start_rules:
                     #   grid.change_all_rules(rule)

                    for rule in best_rules:
                        print(rule)
                        if rule[0] == -1 and rule[1] == -1:
                            grid.change_all_rules([rule[2]])
                        elif rule[0] == -1:
                            grid.change_col_rules(rule[1],[rule[2]])
                        else:
                            if rule[1] == -1:
                                grid.change_row_rules(rule[0],[rule[2]])
                            else:
                                grid.change_cell_rules(rule[0],rule[1],[rule[2]])
                    
                   # grid.add_input("00110"*2)

                    dist = 0.

                    for j in range(len(inputs)):
                        grid.reset()
                        inp = inputs[j]
                        grid.add_input(inp)
                        grid.draw_targets(targets[j])
                        grid.update(0.025)
                        grid.draw_targets(targets[j])
                        output ="0" +grid.cell_states[0] + "0"
                        d = (equi_dist(output, targets[j]))

                        dist += d**2
                        print d, dist

                        pygame.display.update()
                        t = time.time() 
                        while time.time() - t < 0.5:
                            pass


                    print dist

                    curr_rules = copy.deepcopy(best_rules)
                    curr_likelihood = best_likelihood
                    curr_posterior = best_posterior



                elif event.key == pygame.K_f:
                    delay = delay / 2.
                elif event.key == pygame.K_s:
                    delay = delay * 2.



        pygame.display.update()

        count += 1
        #if (grid.cell_states[margin]  != output):

        #output =grid.cell_states[margin]
        #distance = levenshtein(output,target)





if __name__ == "__main__":
    n_cells = 30
    grid_size = int(width/n_cells)
    margin=1
    main(n_cells, grid_size, margin)
