import rclpy
from rclpy.node import Node
import rclpy.qos

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
import numpy as np

import time
import threading

class Turtlebot3Follow(Node):

    def __init__(self):
        super().__init__('turtlebot3_follow_node')

        # definition of publisher and subscriber object to /cmd_vel and /scan 
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 1)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.laser_callback, rclpy.qos.qos_profile_sensor_data)

        self.angle_min = 0
        self.angle_increment = 0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.ranges = []
        # list of distance in angles range
        self.range_view = []
        # list of rays angles in angles range
        self.angle_view = []
        self.stop = False
        timer_period = 0.1  # seconds

        self.timer = self.create_timer(timer_period, self.control_loop)


    # loop each 0.1 seconds
    def control_loop(self):
        msg_pub = Twist()
        msg_pub.linear.x  = self.linear_vel
        msg_pub.angular.z = self.angular_vel
        self.publisher_.publish(msg_pub)

    def stop_robot(self):
        self.stop = True
        
    # called each time new message is published
    def laser_callback(self, msg):

        # stop robot when shutdown node
        if self.stop:
            self.linear_vel = 0.0
            self.angular_vel = 0.0

            msg_pub = Twist()
            msg_pub.linear.x  = 0.0
            msg_pub.angular.z = 0.0
            self.publisher_.publish(msg_pub)
            return
        
        self.range_view.clear()
        self.angle_view.clear()
        
        # check sanity of ranges array
        if self.len_ranges < 100:
            self.linear_vel = 0.0
            self.angular_vel = 0.0
            return


        # Write your code here
        


def main(args=None):
    rclpy.init(args=args)

    turtlebot3_follow_node = Turtlebot3Follow()

    t = threading.Thread(target=rclpy.spin, args=[turtlebot3_follow_node])
    t.start()

    try:
        while rclpy.ok():
            time.sleep(5)
    except KeyboardInterrupt:
        turtlebot3_follow_node.stop_robot()
        time.sleep(0.5)

    # Destroy the node explicitly

    turtlebot3_follow_node.destroy_node()
    rclpy.shutdown()

    t.join()


if __name__ == '__main__':
    main()
