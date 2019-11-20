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

class Image_processing:

    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        # uses an adaptive algorithm to match messages based on their timestamp
        message_filters.ApproximateTimeSynchronizer([
                message_filters.Subscriber('image_topic1', Image),
                message_filters.Subscriber('image_topic2', Image)],
                10, 0.1
            ).registerCallback(self.callback)


    def compute_current_position(self, cv_image1, cv_image2):
        common_image1 = Common_image(cv_image1)
        joint_positions1 = common_image1.compute_joint_positions()
        target_position1 = common_image1.compute_target_center()
        print(joint_positions1)
        print(target_position1)

        common_image2 = Common_image(cv_image2)
        joint_positions2 = common_image2.compute_joint_positions()
        target_position2 = common_image2.compute_target_center()
        print(joint_positions2)
        print(target_position2)

    def callback(self, data1, data2):
        try:
            cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
            cv_image2 = self.bridge.imgmsg_to_cv2(data2, "bgr8")

            self.compute_current_position(cv_image1, cv_image2)

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