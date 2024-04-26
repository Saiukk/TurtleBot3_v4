from re import T
import pandas as pd
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from std_msgs.msg import String
import math

class A_StarPlanner(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.publisher_velocity = self.create_publisher(Twist, '/cmd_vel', 1)
        self.odom_sub = self.create_subscription(Odometry,'/odom', self.odom_callback,10)
        timer_period = 2  # seconds

        self.done_odom = 0
        self.current_angle = 0
        self.current_x = 0
        self.current_y = 0
        self.angle_done = 0
        self.csv_line = 0
        self.csv_file_path = 'new_test_2.csv'
        #initialize
        self.map_positions = pd.read_csv(self.csv_file_path, header=None)
        self.goal_x  = self.map_positions[0][self.csv_line]
        self.goal_y  = self.map_positions[1][self.csv_line]
        self.goal_angle = calculate_angle(self.current_x,self.current_y,self.goal_x,self.goal_y)
        


    def odom_callback(self, odom_msg):
        self.current_pose = odom_msg.pose.pose
        quaternion = (
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        )
        posizione = (

            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_pose.position.z
        )

        #print("The current position is  " + str(posizione))
        #print("Trying to reach: " + str(self.goal_x) + "  " +str(self.goal_y))
        print("Trying to reach: {:.2f} {:.2f}".format(self.goal_x, self.goal_y))
        #print("The current position is {:.2f}".format(posizione))
        print("The current position is ({:.2f}, {:.2f}, {:.2f})".format(posizione[0], posizione[1], posizione[2]))

        l = euler_from_quaternion(quaternion)
        l = [x * (180/3.14) for x in l]
        self.current_angle = l[2]
        self.current_x = posizione[0]
        self.current_y = posizione[1]
        #code to get rid of in case
        if self.current_angle < 0:
            self.current_angle = remap(self.current_angle,-180,0,180,360)
        if self.goal_angle < 0:
            self.goal_angle = remap(self.goal_angle,-180,0,180,360)
            
        angle_difference = self.goal_angle - self.current_angle
        position_difference_x = self.goal_x -self.current_x
        position_difference_y = self.goal_y - self.current_y
        angular_distance = calculate_angular_distance(self.current_angle,self.goal_angle)
        if abs(angle_difference) > 0.2:
            #print('Adjusting to angle: ' + str(self.goal_angle) + ' Current angle is: ' + str(self.current_angle))
            print('Adjusting to angle: {:.2f} Current angle is: {:.2f}'.format(self.goal_angle, self.current_angle))

            if angular_distance >= 0: # was 0
                msg_pub = Twist()
                msg_pub.angular.z = +0.1
                self.publisher_velocity.publish(msg_pub)
            else:
                msg_pub = Twist()
                msg_pub.angular.z = -0.1
                self.publisher_velocity.publish(msg_pub)
        else:
            msg_pub = Twist()
            msg_pub.angular.z = 0.0
            self.publisher_velocity.publish(msg_pub)
            self.angle_done = 1

        if self.angle_done:
            #print('Should adjust speed  ' + str(position_difference_x) + ' ' + str(position_difference_y))
            if abs(position_difference_x) > 0.01 or abs(position_difference_y) > 0.01:
                #print('Adjusting speed  ' + str(position_difference_x) + ' ' + str(position_difference_y))
                msg_pub = Twist()
                msg_pub.linear.x = 0.08
                self.publisher_velocity.publish(msg_pub)
            else:
                msg_pub = Twist()
                msg_pub.linear.x = 0.0
                self.publisher_velocity.publish(msg_pub)
                print('Done')
                self.angle_done = 0
                self.define_new_goal()

    def define_new_goal(self):
        self.csv_line += 1
        self.goal_x  = self.map_positions[0][self.csv_line]
        self.goal_y  = self.map_positions[1][self.csv_line]
        self.goal_angle = calculate_angle(self.current_x,self.current_y,self.goal_x,self.goal_y)
        
def remap(value, from_low, from_high, to_low, to_high):
    from_range = from_high - from_low
    to_range = to_high - to_low
    
    scaled_value = (value - from_low) / from_range
    
    return to_low + (scaled_value * to_range)
        


def calculate_angle(current_x, current_y, goal_x, goal_y):
    # Calculate the angle in radians
    angle_rad = math.atan2(goal_y - current_y, goal_x - current_x)

    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)

    # Ensure the angle is in the range [-180, 180)
    angle_deg = (angle_deg + 180) % 360 - 180

    return angle_deg

    def timer_callback(self):
        print('hello')

def calculate_angular_distance(start_angle, end_angle):
    clockwise_distance = (end_angle - start_angle) % 360
    counterclockwise_distance = (start_angle - end_angle) % 360

    if clockwise_distance <= counterclockwise_distance:
        return clockwise_distance
    else:
        return -counterclockwise_distance  # Negative value indicates counterclockwise direction


def main(args=None):
    rclpy.init(args=args)
    A_star_planner = A_StarPlanner()
    rclpy.spin(A_star_planner)
    A_star_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()