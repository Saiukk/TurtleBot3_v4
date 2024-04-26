class Node():
    def __init__(self,father,gcost,hcost,flag,x,y):
        self.parent = father #parent of the node
        self.x = x #coordinate of the x
        self.y = y #coordinate of the y
        self.gcost = gcost 
        self.hcost = hcost
        self.fcost = None #set to none since we don't know the h yet
        self.flag = flag # start goal walkable obstacle
        
    # define f cost
    def f_cost(self):
        self.fcost = self.gcost + self.hcost
    
    # debug print    
    def print_coordinates(self):
        if self.fcost != None:
            print('Node at: (' + str(self.x) + ',' + str(self.y) + ')' + 'Cost g and h and f    ' + str(self.gcost)+ ', ' +str(self.hcost) +',  ' + str(self.fcost))