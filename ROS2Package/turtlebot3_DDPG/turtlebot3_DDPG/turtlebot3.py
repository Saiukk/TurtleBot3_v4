import copy
import math
import time

import numpy as np

from nav_msgs.msg import Odometry
import rclpy
from geometry_msgs.msg import Twist, Point
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
import rclpy.qos
import tf_transformations
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from math import pi


class TurtleBot3():

    def __init__(self):

        #qos = QoSProfile(depth=10)
        # self.node = rclpy.create_node('turtlebot3_DDQN_node')
        self.lidar_msg = LaserScan()
        self.odom_msg = Odometry()
        # set your desired goal: 
        self.goal_x, self.goal_y = -1.959, -0.289  #-0.78 , -1.1 #-0.777, 1.176#z  # -0.777, 1.176 ##    # # this is for simulation change for real robot

        # linear velocity is costant set your value
        self.linear_velocity = 0.2  # to comment
        self.distanceNormFact = 3
        # ang_vel is in rad/s, so we rotate 5 deg/s [action0, action1, action2]
        self.angular_velocity = [0.0, -2.20, 2.20]  # to comment

        # self.r = rclpy.spin_once(self.node,timeout_sec=0.25)

        print("Robot initialized")

    def SetLaser(self, msg):
        self.lidar_msg = msg

    def SetOdom(self, msg):
        self.odom_msg = msg

    def stop_tb(self):
        self.pub.publish(Twist())

    def get_odom(self):

        # read odometry pose from self.odom_msg (for domuentation check http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)
        # save in inpot variable the position
        # save in rot variable the rotation

        point = self.odom_msg.pose.pose.position

        # Axis reversed
        new_point = copy.deepcopy(point)  # Copy also the deep structure

        new_point.x = - point.y
        new_point.y = 0.0
        new_point.z = point.x

        rot = self.odom_msg.pose.pose.orientation

        self.rot_ = tf_transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])

        return new_point, (1 + (np.rad2deg(self.rot_[2]) / 180)) / 2

    def get_scan(self):

        ranges = []
        scan_val = []

        # read lidar msg from self.lidar_msg and save in scan variable
        len_ranges = len(self.lidar_msg.ranges)
        angle_min = self.lidar_msg.angle_min
        angle_increment = self.lidar_msg.angle_increment

        for i in range(len_ranges):  # cast limit values (0, Inf) to usable floats
            if self.lidar_msg.ranges[i] == float('Inf') or math.isnan(self.lidar_msg.ranges[i]) or \
                    self.lidar_msg.ranges[i] > 1.0:
                self.lidar_msg.ranges[i] = 1.0
            # self.lidar_msg.ranges[i] /= 5.5

        # get the rays like training if the network accept 3 (7) rays you have to keep the same number
        # I suggesto to create a list of desire angles and after get the rays from the ranges list
        # save the result in scan_val list

        for (i, range_record) in enumerate(self.lidar_msg.ranges):
            ranges.append({
                "angle": angle_min + (i * angle_increment) if (i < len_ranges / 2) else angle_min + (
                        i * angle_increment) - 6.28,
                "value": range_record
            })
        sorted_ranges = sorted(ranges, key=lambda x: x["angle"])

    # SORTING LIST
        print(len(sorted_ranges))

    # MEAN LIDAR
        #180 90 270
        mean = 0
        count = 0
        for i in range(90, 271):
            mean += sorted_ranges[i]['value']

            if i == 90 or i == 180 or i == 270:
                scan_val.append(sorted_ranges[i]['value'])
                mean = 0
            elif abs(count) % 30 == 0:
                mean /= 30
                scan_val.append(mean)
                mean = 0

            count += 1

        scan_val.reverse()

        return scan_val



    def get_goal_info(self, tb3_pos):

        # # compute distance euclidean distance use self.goal_x/y pose and tb3_pose.x/y
        # # compute the heading using atan2 of delta y and x
        # # subctract the actual robot rotation to heading
        # # save in distance and heading the value
        # #
        # # print('BOT Pos x -> ', tb3_pos.x , 'Pos z -> ', tb3_pos.z, 'Pos y -> ', tb3_pos.y)
        #
        distance_y = self.goal_y - tb3_pos.z
        distance_x = self.goal_x - tb3_pos.x

        # EUCLIDEAN FUNCTION
        distance = math.sqrt(((distance_x) ** 2) + ((distance_y) ** 2))

        # Calculate Heading
        '''
        angle = math.atan2(distance_y, distance_x) # angle between target and position
        print("Angle -> ", np.rad2deg(angle))
        angle -= pi/2
        angle = self.pi_T(angle)
        print("\nAngle -90 -> ", np.rad2deg(angle))

        heading = self.pi_T(angle - self.pi_T(self.rot_[2])
        print("\nHead -90 -> ", np.rad2deg(heading))
'''

        # print("\nHeading final -90 -> ", np.rad2deg(heading))
        #
        # print('Heading -> ', heading)
        # heading = heading - self.rot_[2]
        #
        # print('\n(X, Y --> ', self.goal_x, ', ', self.goal_y, ')', '\nDIST --> ')
        #
        # # we round the distance dividing by 2.8 under the assumption that the max distance between
        # # two points in the environment is approximately 3.3 meters, e.g. 3m
        # # return heading in deg
        # return distance, np.rad2deg(heading) / 180

        yaw = self.pi_domain(np.rad2deg(self.rot_[2]))
        print("yaw: ", yaw)
        angleInDegrees = np.rad2deg(np.arctan2(self.goal_y - tb3_pos.z, self.goal_x - tb3_pos.x))
        angleInDegrees = self.pi_domain(angleInDegrees - 90)
        print("angle deg: ", angleInDegrees)
        heading = -self.pi_domain(angleInDegrees - yaw)
        print(f"heading: {heading}\n")
        print(f"GET GOAL DIST: {distance}, Heading: {heading}")

        # we round the distance dividing by 2.8 under the assumption that the max distance between
        # two points in the environment is approximately 3.3 meters, e.g. 3m
        # return heading in deg
        return distance / self.distanceNormFact, 0.5 + (heading / 360)  # (np.rad2deg(heading) / 360)

    def pi_domain(self, a):
        #print("A Pre: ", a)
        # modulo
        a = a - int(a / 360) * 360  # if a > 360 restart from 0

        #print("A Post: ", a)
        if a > 180:
            a = a - 360
            print("A Neg: ", a)
        if a < -180:
            a = a + 360
            print("A Pos: ", a)
        print("A out: ", a)
        return a

    def pi_T(self, heading):
        heading = heading - int(heading / 2 * pi) * 2 * pi
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        return heading

    def move(self, action, pub):
        # stop robot
        if action == -1:
            pub.publish(Twist())
        else:
            # check action 0: move forward 1: turn left 2: turn right
            # save the linear velocity in target_linear_velocity
            # save the angular velocity in target_angular_velocity

            twist = Twist()
            target_linear_velocity = self.linear_velocity
            target_angular_velocity = self.angular_velocity

            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0

            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0

            # Move Forward
            if action == 0:
                twist.linear.x = target_linear_velocity

            # Turn left
            if action == 1:
                twist.angular.z = target_angular_velocity[1]

            # Turn Right
            if action == 2:
                twist.angular.z = target_angular_velocity[2]

            pub.publish(twist)
