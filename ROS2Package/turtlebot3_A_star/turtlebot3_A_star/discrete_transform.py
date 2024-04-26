#Code for discretization

#Add a new tuple
def add_tuple_to_rough(rough, new_tuple):
    rough.append(new_tuple)

#Reasoning about this is in the readme file, however to put it shortly: maps a continuous value into a discrete one   
#horizontal mapping
def assign_value_horizontal(number):
    lower_bound_positive = 0.5
    lower_bound_negative = -0.5001
    interval = 1.0

    if number >= lower_bound_positive:
        #print(int((number - lower_bound_positive) // interval))
        result = int((number - lower_bound_positive) // interval) + 1
    elif number <= lower_bound_negative:
        number *= -1
        result = int((number - (-1 * lower_bound_negative)) // interval) + 1
        result *= -1
    else:
        result = 0
    #print(result)
    return result

#vertical mapping 
def assign_value_vertical(number):
    lower_bound_positive = 0.3125 
    lower_bound_negative = -0.6876
    interval = 1.0

    if number >= lower_bound_positive:
        result = int((number - lower_bound_positive) // interval) + 1
    elif number <= lower_bound_negative:
        number *= -1
        result = int((number - (-1 * lower_bound_negative)) // interval) + 1
        result *= -1
    else:
        result = 0
    #print(result)
    return result
    
    
def update_obstacles(obstacles, new_obstacle_tuple):
    #rotated = 0 if obstacle is vertical, 1 if obstacle is horizontal
    x, scale_x, y, scale_y,rotated = new_obstacle_tuple[:5]
    y *= -1
    x_off = 0.5 * scale_x
    y_off = 0.5 * scale_y
    
    if rotated == 1:
        x_off, y_off = y_off,x_off
    #print("x,y off")
    #print(x_off, y_off)
    lower_x = assign_value_horizontal((x - x_off) / 0.22)
    upper_x = assign_value_horizontal((x + x_off) / 0.22)
    lower_y = assign_value_vertical((y - y_off) / 0.16)
    upper_y = assign_value_vertical((y + y_off) / 0.16)
    for x_value in range(lower_x, upper_x+1):
        for y_value in range(lower_y,upper_y+1):
            obstacles.append((x_value,y_value))
    pass

def update_start_and_goal(sg, sg_tupla):
    #rotated = 0 if obstacle is vertical, 1 if obstacle is horizontal
    x,y = sg_tupla[:2]
    y *= -1
    x = assign_value_horizontal(x/0.22)
    y = assign_value_vertical(y/0.16)
    sg.append((x,y))
    pass
