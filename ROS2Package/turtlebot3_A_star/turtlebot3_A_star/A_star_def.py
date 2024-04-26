#Library import, node contains Node class
import node
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import cv2
import os
from discrete_transform import *

#fist case start and goal coordinates
start_and_goal_unity = [(0,0),(-1.5,-2.5)] #x,y following unity reference frame (-z is x and x is y ), the code will take care of the -z to +z later
#second and third case start and goal coordinates
#start_and_goal_unity = [(-1.1,-0.48), (-0.22,-2.72)]

#initialization of the arrays which will contain the discrete coordinates
start_and_goal = []
obstacles = []
# tuple of  obstacle data are put in this way: (x,scale_x,y,scale_y,1 if horizontal else 0 if vertical), x is equal to z in unity and y is equal to x in unity, code will take care of the -z error 

#first case obstacles positions
obstacles_data = [(-0.22,0.1,-1,0.3,0),(-1.326,0.1,-1,0.3,1),(-0.002,0.1,-0.313,0.3,1),(-0.684,0.1,-1,0.6,1),(-0.819,0.1,-1.813,0.3,0),(-1.283,0.1,-1.813,0.6,0),(-2.199,0.1,-1.813,0.3,0),(-2.663,0.1,-1.813,0.6,0),(-2.33,0.1,-0.267,0.3,1),(-1.688,0.1,0.267,0.6,1),(-1.326,0.1,-2.267,0.3,1),(-0.684,0.1,-2.267,0.6,1),(-0.819,0.1,-3.08,0.3,0),(-1.283,0.1,-3.08,0.6,0),(-2.199,0.1,-3.08,0.3,0)]
#second case obstacles positions
#obstacles_data = [(0.037,0.1,-1,0.3,0),(-1.129,0.1,-0.023,3.8,1),(-2.314,0.1,-1.796,3,0),(-0.777,0.1,-1.163,0.3,0),(-0.462,0.1,-1.609,0.3,0),(-1.209,0.1,-1.595,0.3,0),(-2.023,0.1,-1.758,0.3,0),(-1.708,0.1,-2.204,0.3,0),(-0.182,0.1,-2.091,0.3,1),(-0.777,0.1,-2.464,0.3,1),(-0.462,0.1,-2.91,0.3,1),(-1.209,0.1,-2.234,0.3,0),(-0.003,0.1,-0.648,0.3,1),(-0.502,0.1,-1.257,0.3,1),(-1.472,0.1,-0.893,0.3,1),(-1.146,0.1,-0.862,0.3,0)]
#third case obstacles positions
#obstacles_data = [(0.037,0.1,-1,0.3,0),(-1.129,0.1,-0.023,3.8,1),(-2.314,0.1,-1.796,3,0),(-0.777,0.1,-1.163,0.3,0),(-0.462,0.1,-1.609,0.3,0),(-1.209,0.1,-1.595,0.3,0),(-2.023,0.1,-1.758,0.3,0),(-1.708,0.1,-2.204,0.3,0),(-0.182,0.1,-2.091,0.3,1),(-0.777,0.1,-2.464,0.3,1),(-0.462,0.1,-2.91,0.3,1),(-1.209,0.1,-2.234,0.3,0),(-0.003,0.1,-0.648,0.3,1),(-0.817,0.1,-0.811,0.3,1),(-0.502,0.1,-1.257,0.3,1),(-1.472,0.1,-0.893,0.3,1),(-0.687,0.1,-2.091,0.3,1),(0.201,0.1,-1.744,0.7,0)]

#discretization of the obstacles and start,goal positions
for tupla in obstacles_data:
    update_obstacles(obstacles,tupla)

for tupla_sg in start_and_goal_unity:
    update_start_and_goal(start_and_goal,tupla_sg)


#offset adjustmets -> make the code work only with positive values by shifting all coordinates
values = start_and_goal + obstacles #create an array with all coordinates

#get the min values on x and y
offset_x = min(values, key=lambda x: x[0])[0] 
offset_y = min(values, key=lambda x: x[1])[1]
#if an offset is negative then I shift all the points by that offset to get a positive value
adjusted_values = [(x - offset_x if offset_x <= 0 else x, y - offset_y if offset_y <= 0 else y) for x, y in values]
#assign the new coordinates
start = adjusted_values[0]
goal = adjusted_values[1]
obstacles = adjusted_values[2:]


# Find the maximum x and y values
max_x = max(adjusted_values, key=lambda point: point[0])[0]
max_y = max(adjusted_values, key=lambda point: point[1])[1]

#Map sizes -> got from the maximum values (+1 cuz 0 is included in the coordinates)
size_map_x = max_x + 1
size_map_y = max_y + 1

#initialize the map as empty nodes -> for more details check node class
map = np.empty([size_map_x,size_map_y], dtype = node.Node)

#initialize sets for the A* algorithm
closed_set = []
open_set = []
#costants for attractive force and repulsive force and do, size of repulsive field,weight if you move oriz/ver or diagonally (orizontally is prefered)
ka = 15 #attractive constant
kb = 150 #repulsive constant
do = 3 # radius effect of repulsive force
simple_weight = 5 # robot prefers to move horizontally if has the same weight (less travel time)
diagonal_weight = 7 # weight to move vertically

#Defining the node type in the empty map

#start node is gonna have g cost = 0 others infinite
#the h cost are gonna be filled later when having all the nodes

for i in range(size_map_x):
    for j in range(size_map_y):
        
        if (i,j) == start:
            map[i][j] = node.Node(None,0,0,'start',i,j)
        elif (i,j) == goal:
            map[i][j] = node.Node(None,999999999999,0,'goal',i,j)
        elif (i,j) in obstacles:
            map[i][j] = node.Node(None,999999999999,0,'obstacle',i,j)
        else:
            map[i][j] = node.Node(None,999999999999,0,'walkable',i,j)
            


# code for image visualization
image = plt.gca()
image.set_xlim([0,size_map_x])
image.set_ylim([0,size_map_y])

#Assigning colors to tiles based on the type:
#obstacle = gray
#start = blue
#goal = red
for i in range(size_map_x):
    for j in range(size_map_y):
        if map[i][j].flag == 'obstacle':
            rec = Rectangle((i,j), width= 1, height= 1, facecolor = 'gray')
            image.add_patch(rec)
        elif map[i][j].flag == 'start': 
            rec = Rectangle((i,j),width=1,height=1,facecolor = 'b')
            image.add_patch(rec)
        elif map[i][j].flag == 'goal':
            rec = Rectangle((i,j),width=1,height = 1, facecolor = 'r')
            image.add_patch(rec)    
        else:
            rec = Rectangle((i,j), width= 1, height= 1, edgecolor = 'gray', facecolor = 'w')
            image.add_patch(rec)
plt.axis('equal')
plt.axis('off')
plt.tight_layout()

#function to save images in time order
def SaveImage(plt):
    millis = int(round(time.time() * 1000))
    filename = './images/'+str(millis)+'.png'
    plt.savefig(filename)


# Get the path by tracing back the nodes from the goal
def reconstruct_path(current):
    total_path = []
    total_path.append(current) #add the last node
    
    #Adding nodes by tracing back the parent, if I have no more parent it means i reached the starting node
    while current.parent!= None:
        current = current.parent
        total_path.append(current)
        
        
    # Visual representation of the reconstructed path (green)
    for element in total_path:
        rec = Rectangle((element.x, element.y),width = 1, height= 1, color = 'g')
        image.add_patch(rec)
        rx, ry = rec.get_xy()
        cx = rx + rec.get_width()/2.0
        cy = ry + rec.get_height()/2.0
        image.annotate(str(int(element.fcost)), (cx, cy), color='white', weight='bold', fontsize=5, ha='center', va='center')
        plt.draw()
        SaveImage(plt)
        
    return total_path


#attractive force based on paper formulas, linear based on distance
def attractive_force(map,i,j):
    # TODO
    return 0

#repulsive force based on paper formulas
def repulsive_force(map,i,j):
    # TODO
    return 0

#getting the h cost of a tile; this is the heuristic part of A*
def h_cost(map,i,j):
    attrattiva = attractive_force(map,i,j)
    repulsiva = repulsive_force(map,i,j)
    
    #simply sum the forces (both norms)
    hcost = attrattiva + repulsiva
    return hcost

def test_diagonal(x1,x2,y1,y2):
    to_evaluate_1 = map[x1][y1]
    to_evaluate_2 = map[x2][y2]
    if to_evaluate_1.flag == 'walkable' and to_evaluate_2.flag == 'walkable':
        return True 
    return False

    
#checks if the node taken in consideration should be evaluated
#the node must be start,goal and walkable
#the node must not be an obstacle
#the node must be into the map boundaries

def test(x,y):    
    if x>=0 and x<size_map_x and y>=0 and y<size_map_y:
        to_evaluate = map[x][y]
        if to_evaluate.flag == 'start' or to_evaluate.flag == 'goal' or to_evaluate.flag == 'walkable':
            return True 
        return False
    return False

        
#Node evaluation (A* algorithm)       
def evaluate_neighbor(x,y,peso,current):
    
    evaluating = map[x][y] # take the node to evaluate
    

    # if it is in the closed set I've already evaluated it, can skip
    if evaluating in closed_set:
        return 
    
    tentative_g_score = current.gcost + peso #assign the new g_score (depends on the direction I'm coming from)
    
    # Add it to open set if not already there
    if evaluating not in open_set:
        open_set.append(evaluating)
        tentative_is_better = True #It's better since I am not substituting anything
    # It's in open set but the cost is better
    elif tentative_g_score < evaluating.gcost:
        tentative_is_better = True
    # It's in open set but the cost is worse
    else:
        tentative_is_better = False
    
    #If it is better then I change the node to evaluate to the one with the lower cost and assing the node where I come from as a parent
    if tentative_is_better:
        evaluating.parent = current
        evaluating.gcost = tentative_g_score # assigning g cost (travel cost)
        evaluating.hcost = h_cost(map,evaluating.x,evaluating.y) # assigning h cost (forces cost)
        evaluating.f_cost() # node function, just sum g and h costs
        
    if evaluating.flag == 'goal':
        evaluating.parent = current
        evaluating.gcost = tentative_g_score # assigning g cost (travel cost)
        evaluating.hcost = h_cost(map,evaluating.x,evaluating.y) # assigning h cost (forces cost)
        evaluating.fcost = -1 # node function, just sum g and h costs

    
#code to create video for visualization
def MakeVideo():
    image_folder = './images'
    video_name = 'video.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 15, (width,height)) 
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        
    cv2.destroyAllWindows()
    video.release()
    
#A star code 
def a_star(map,start,goal,obstacles):
    
    #Add the starting node and get its h cost
    open_set.append(map[start[0]][start[1]])
    open_set[0].h = h_cost(map,start[0],start[1])
    open_set[0].f_cost()
    
    # Evaluating open set
    while len(open_set) > 0:
        
        #select from open set the one with the lowest fcost
        current = open_set[0]
        min_f = current.fcost
        for element in open_set:
            if element.fcost < min_f:
                min_f = element.fcost
                current = element
        # Images code                
        rec = Rectangle((current.x,current.y),width=1,height=1,color = 'c')
        image.add_patch(rec)
        rx, ry = rec.get_xy()
        cx = rx + rec.get_width()/2.0
        cy = ry + rec.get_height()/2.0
        image.annotate(str(int(current.fcost)), (cx, cy), color='black', weight='bold', fontsize=6, ha='center', va='center')
        SaveImage(plt)
        
        #If I've reached the end I trace back to the beginning
        if current.flag == 'goal':
            return reconstruct_path(current)
        
        #Remove the node from open set and add it to the closed set
        open_set.remove(current)
        closed_set.append(current)
        
        #evaluation of all the neighbours of current
        #get the coordinates of current
        index_x = current.x
        index_y = current.y
        #for every node -> evaluate if the following conditions are satisfied:
        #test -> it is not an obstacles and it is between map boundaries
        #test_diagonal -> the nodes which I need to pass through are free
        
        #top right node
        if test(index_x + 1, index_y + 1) == True and test_diagonal(index_x + 1, index_x,index_y,index_y+1)==True:
            evaluate_neighbor(index_x+1,index_y+1,diagonal_weight,current)
        

        #right node
        if test(index_x + 1, index_y)== True:
            evaluate_neighbor(index_x + 1, index_y,simple_weight,current)

        #bottom right node
        if test(index_x + 1, index_y - 1)== True and test_diagonal(index_x + 1, index_x,index_y,index_y-1)== True:
            evaluate_neighbor(index_x + 1, index_y - 1,diagonal_weight,current)

        #bottom node
        if test(index_x, index_y - 1)== True:
            evaluate_neighbor(index_x, index_y - 1,simple_weight,current)
  
        #bottom left node
        if test(index_x - 1, index_y - 1)== True and test_diagonal(index_x,index_x-1,index_y-1,index_y)== True:
            evaluate_neighbor(index_x - 1, index_y - 1,diagonal_weight,current)
 
        #left node
        if test(index_x - 1, index_y)== True:
            evaluate_neighbor(index_x - 1, index_y,simple_weight,current)
  
        #top left node
        if test(index_x - 1, index_y + 1)== True and test_diagonal(index_x,index_x-1,index_y + 1, index_y) == True:
            evaluate_neighbor(index_x - 1, index_y + 1,diagonal_weight,current)
   
        #top node
        if test(index_x, index_y + 1)== True:   
            evaluate_neighbor(index_x, index_y + 1,simple_weight,current)
  
    #If there is no path I signal the error
    print('Failed')
    return 'Error'


solution = [] # initialize solution array
solution = a_star(map,start,goal,obstacles) # get the solution array

#Print the solution in reverse order and in continuous domain  (0.22 is x units, 0.16 is y units -> turtlebot dimensions)
#Copy those values in the csv file in the moving code (map_following.py)
for nodo in solution[::-1]:
    nodo.x += offset_x
    nodo.y += offset_y
    print(str((nodo.x)* 0.22) + ',' + str(nodo.y*0.16))

#Add the goal
goal_x = goal[0] + offset_x
goal_y = goal[1] + offset_y

#comment to not get the video
MakeVideo()



