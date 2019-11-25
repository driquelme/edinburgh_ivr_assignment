#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from common_image import Common_image
from scipy.optimize import least_squares
from numpy.linalg import norm
from joint_state import JointState
from kinematics import Kinematics


class Image_processing:

    def __init__(self):

        # We compute the following values only once
        # self.meter_to_pixel_factor = -1
        self.origin_x = -1
        self.origin_y = -1
        self.origin_z = -1

        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)

        self.target_pub = rospy.Publisher("target_position", Float64MultiArray, queue_size=1)
        self.kinematics_pub = rospy.Publisher("kinematics", Float64MultiArray, queue_size=1)
        self.red_pub = rospy.Publisher("red", Float64MultiArray, queue_size=1)
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.joint_state = JointState()

        # uses an adaptive algorithm to match messages based on their timestamp
        message_filters.ApproximateTimeSynchronizer([
            message_filters.Subscriber('/camera1/robot/image_raw', Image),
            message_filters.Subscriber('/camera2/robot/image_raw', Image)],
            10, 0.1
        ).registerCallback(self.callback)

    def compute_joint_state(self, cv_image1, cv_image2):

        self.joint_state.cv_image1 = cv_image1
        self.joint_state.cv_image2 = cv_image2

        kinematics = Kinematics()

        result = self.joint_state.compute_joint_state(False)

        if (result is not None):
            target_position = self.joint_state.compute_target_position(False)
            if (target_position is not None):
                target_position_meters = target_position * self.joint_state.meter_to_pixel_factor

                # Publish target position
                self.target_pub.publish(Float64MultiArray(data=target_position_meters))

                k = kinematics.kinematics(self.joint_state)

                # print("K")
                # print(k[0:3,-1:])
                # print("red")
                # print(self.joint_state.red_pos * self.joint_state.meter_to_pixel_factor)

                # Publish Kinematics
                self.kinematics_pub.publish(Float64MultiArray(data=k[0:3,-1:] ))

                # Publish red joint position
                self.red_pub.publish(Float64MultiArray(data=self.joint_state.red_pos * self.joint_state.meter_to_pixel_factor))

                new_joint_state = self.joint_state.control_closed(True)
                if (new_joint_state is not None):
                    self.robot_joint1_pub.publish(new_joint_state[0])
                    self.robot_joint2_pub.publish(new_joint_state[1])
                    self.robot_joint3_pub.publish(new_joint_state[2])
                    self.robot_joint4_pub.publish(new_joint_state[3])


                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


                # cv2.circle(cv_image1, (int(k[1, 3] / self.joint_state.meter_to_pixel_factor + self.joint_state.origin_pos[1]),
                #                        int(-k[2, 3] / self.joint_state.meter_to_pixel_factor + self.joint_state.origin_pos[2])),
                #            10, (0, 0, 255), thickness=4, lineType=8)
                #
                # cv2.circle(cv_image1, (int(joint_state.origin_pos[1]),
                #                        int(joint_state.origin_pos[2])),
                #            18, (0, 255, 255), thickness=4, lineType=8)
                # cv2.circle(cv_image1, (int(joint_state.blue_abs_pos[1]),
                #                        int(joint_state.blue_abs_pos[2])),
                #            16, (255, 0, 0), thickness=4, lineType=8)

                # cv2.imshow('k1', cv_image1)
                # cv2.waitKey(1)
                #
                # cv2.circle(cv_image2, (int(k[0, 3] / self.joint_state.meter_to_pixel_factor + self.joint_state.origin_pos[0]),
                #                        int(-k[2, 3] / self.joint_state.meter_to_pixel_factor + self.joint_state.origin_pos[2])),
                #            10, (0, 0, 255), thickness=4, lineType=8)


                # cv2.circle(cv_image1,
                #            (int(joint_positions1['green'][0]), int(joint_positions1['green'][1])), 10,
                #            (255, 0, 255), thickness=2, lineType=8)
                # cv2.circle(cv_image1,
                #            (int(joint_positions1['blue'][0]), int(joint_positions1['blue'][1])), 10,
                #            (255, 0, 255), thickness=2, lineType=8)
                # cv2.circle(cv_image2,
                #            #(int(joint_state.red_abs_pos[0]), int(joint_state.red_abs_pos[2])), 5,
                #            (int(3 / self.joint_state.meter_to_pixel_factor + self.joint_state.origin_pos[0]),
                #             int(-5 / self.joint_state.meter_to_pixel_factor + self.joint_state.origin_pos[2])), 10,
                #            (0, 10, 205), thickness=2, lineType=8)

                # cv2.imshow('k2', cv_image2)
                # cv2.waitKey(1)
            else:
                print("Skipping")
        else:
            print("Skipping")

    def callback(self, data1, data2):
        try:
            cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
            cv_image2 = self.bridge.imgmsg_to_cv2(data2, "bgr8")

            self.compute_joint_state(cv_image1, cv_image2)
            #self.compute_current_position2(cv_image1, cv_image2)

        except CvBridgeError as e:
            print(e)


# call the class
def main(args):
    ic = Image_processing()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
