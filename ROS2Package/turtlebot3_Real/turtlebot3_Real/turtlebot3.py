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

    def __init__(self, point_x, point_y):

        #qos = QoSProfile(depth=10)
        #self.node = rclpy.create_node('turtlebot3_DDQN_node')
        self.lidar_msg = LaserScan()
        self.odom_msg = Odometry()
        self.validationStart = False
        self.count = 0
        ## HARD POSITION

        start_position = self.get_odom()[0]
        #start_position.x = -0.2 #Da commentare per il test in lab (get_odom prende pos relativa alla base)
        #start_position.z = -1.1

        # set your desired goal:
        # Se non va start_position.z prova con start_position.y in real dovrebbe cambiare
        self.goal_x, self.goal_y = point_x, point_y # this is for simulation change for real robot
        # linear velocity is costant set your value
        self.linear_velocity = 0.1  # to comment
        self.distanceNormFact = 3
        # ang_vel is in rad/s, so we rotate 5 deg/s [action0, action1, action2]
        self.angular_velocity = [0.0, -1.20, 1.20]  # to comment

        # self.r = rclpy.spin_once(self.node,timeout_sec=0.25)

        print("Robot initialized")

    def SetLaser(self, msg):
        self.lidar_msg = msg

    def SetOdom(self, msg):
        self.odom_msg = msg

        if self.odom_msg.pose.pose.position.x != 0 and self.odom_msg.pose.pose.position.y != 0 and self.count == 0:
            self.validationStart = True
            self.goal_x += msg.pose.pose.position.x
            self.goal_y += msg.pose.pose.position.y
            self.count += 1
            print("Goal X: ", self.goal_x, "Goal Y: ", self.goal_y)
            time.sleep(2)

    def stop_tb(self):
        self.pub.publish(Twist())

    def get_odom(self):

        # read odometry pose from self.odom_msg (for domuentation check http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)
        # save in inpot variable the position
        # save in rot variable the rotation

        point = self.odom_msg.pose.pose.position
        print("POSIZIONE ORA: ", point)

        # Axis reversed
        new_point = copy.deepcopy(point)  # Copy also the deep structure

        # Change for simulation
        #new_point.x = - point.y
        #new_point.y = 0.0
        #new_point.z = point.x

        rot = self.odom_msg.pose.pose.orientation
        print('Orientation ->', rot)

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
            if i % 3 == 0:
                ranges.append({
                    "angle": angle_min + (i * angle_increment) if (i < len_ranges / 2) else angle_min + (
                            i * angle_increment) - 6.28,
                    "value": range_record
                })
        sorted_ranges = sorted(ranges, key=lambda x: x["angle"])

    # SORTING LIST
        #print((sorted_ranges))
        print(len(sorted_ranges))

        if(len(sorted_ranges) == 0):
            for i in range(7):
                scan_val.append(0.1)

            return scan_val

    # MEAN LIDAR
        #180 90 270
        mean = 0
        count = 0

        # Angoli per la simulazione (lidar posizionato con vista frontale)
        #start_angle = 90
        #end_angle = 270

        # Angoli per real (lidar posizionato con vista laterale sinistra)
        start_angle = 180
        end_angle = 360
        mean_angle = (start_angle + end_angle) / 2

        sorted_ranges.append({
            "angle": 361,
            "value": sorted_ranges[0]['value']})

        for i in range(start_angle, end_angle + 1):
            mean += sorted_ranges[i]['value']

            if i == start_angle or i == mean_angle or i == end_angle:
                scan_val.append(sorted_ranges[i]['value'])
                mean = 0
            elif abs(count) % 30 == 0:
                mean /= 30
                scan_val.append(mean)
                mean = 0

            count += 1


        print('Laser -->', scan_val)

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
        distance_y = self.goal_y - tb3_pos.y             # z con unity
        distance_x = self.goal_x - tb3_pos.x

        # EUCLIDEAN FUNCTION
        distance = math.sqrt(((distance_x) ** 2) + ((distance_y) ** 2))


        yaw = self.pi_domain(np.rad2deg(self.rot_[2]))
        #print("yaw: ", np.rad2deg(self.rot_[2]))
        angleInDegrees = np.rad2deg(np.arctan2(self.goal_y - tb3_pos.y, self.goal_x - tb3_pos.x))     # tb3pos.z con unity
        angleInDegrees = self.pi_domain(angleInDegrees) #-90
        #print("angle deg: ", angleInDegrees)
        heading = -self.pi_domain(angleInDegrees - yaw)
        #print(f"heading: {heading}\n")
        print(f"GET GOAL DIST: {distance/self.distanceNormFact}, Heading: {heading}")

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
            #print("A Neg: ", a)
        if a < -180:
            a = a + 360
            #print("A Pos: ", a)
        #print("A out: ", a)
        return a

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

            if self.validationStart:
                pub.publish(twist)
