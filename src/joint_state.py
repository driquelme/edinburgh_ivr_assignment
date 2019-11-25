import cv2
import numpy as np
from common_image import Common_image
from numpy.linalg import norm
from scipy.optimize import least_squares
from kinematics import Kinematics
import rospy

class JointState():

    def __init__(self):
        #self.cv_image1 = cv_image1
        #self.cv_image2 = cv_image2
        self.meter_to_pixel_factor = -1
        self.initial_angles = [0, 0, 0, 0]

        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')

        self.kinematics = Kinematics()

        print("Init JointState")

    def compute_joint_state(self, debug=False):
        # Detect objects from using images
        self.common_image1 = Common_image(self.cv_image1)
        self.common_image2 = Common_image(self.cv_image2)

        joint_positions1 = self.common_image1.compute_joint_positions()
        joint_positions2 = self.common_image2.compute_joint_positions()

        if (joint_positions1 is None or joint_positions2 is None):
            return None

        # if (debug):
        # self.common_image1.compute_blob_center(self.common_image1.lower_green, self.common_image1.upper_green, debug=True)
        # common_image2.compute_blob_contours(common_image2.lower_green, common_image1.upper_green, debug=True)

        # # # # # # # # # # # # # # #
        #
        # Fix occlusion issues
        #
        joint_positions1, joint_positions2 = self.handle_occlusion(joint_positions1, joint_positions2)

        # # # # # # # # # # # # # # #
        #
        # Compute positions
        #
        self.origin_pos = self.compute_joint_position(joint_positions1['yellow'], joint_positions2['yellow'])

        self.blue_abs_pos = self.compute_joint_position(joint_positions1['blue'], joint_positions2['blue'])
        self.blue_pos = self.transform_to_relative_pos(self.blue_abs_pos, self.origin_pos)

        self.green_abs_pos = self.compute_joint_position(joint_positions1['green'], joint_positions2['green'])
        self.green_pos = self.transform_to_relative_pos(self.green_abs_pos, self.origin_pos)

        self.red_abs_pos = self.compute_joint_position(joint_positions1['red'], joint_positions2['red'])
        self.red_pos = self.transform_to_relative_pos(self.red_abs_pos, self.origin_pos)

        # # # # # # # # # # # # # # #
        #
        # Compute distances
        #
        yellow_blue_distance = self.compute_distance_between_joints(self.origin_pos, self.blue_abs_pos)
        blue_green_distance = self.compute_distance_between_joints(self.blue_abs_pos, self.green_abs_pos)
        green_red_distance = self.compute_distance_between_joints(self.green_abs_pos, self.red_abs_pos)


        # # # # # # # # # # # # # # #
        #
        # Compute pixel to meter factor
        #
        if (self.meter_to_pixel_factor==-1):
            self.meter_to_pixel_factor = self.common_image1.compute_meter_to_pixel_factor(self.origin_pos,
                                                                                      self.blue_abs_pos, 2.0)

        if (debug):
            print("\nmeter_to_pixel_factor")
            print(self.meter_to_pixel_factor)

        self.yellow_blue_distance_meters = yellow_blue_distance * self.meter_to_pixel_factor
        self.blue_green_distance_meters = blue_green_distance * self.meter_to_pixel_factor
        self.green_red_distance_meters = green_red_distance * self.meter_to_pixel_factor

        # # # # # # # # # # # # # # #
        #
        # Compute angles
        #
        self.joint_angles = self.calculate_angles()
        if (debug):
            print("\nAngles")
            print(self.joint_angles)

        return "Done"

    def handle_occlusion(self, joint_positions1, joint_positions2):
        if (joint_positions1['green'] is None):
            print("joint_positions1 occluded")
            joint_positions1['green'] = [0, 0]
            joint_positions1['green'][0] = joint_positions1['red'][0]
            joint_positions1['green'][1] = joint_positions2['green'][1]

        if (joint_positions2['green'] is None):
            print("joint_positions2 occluded")
            joint_positions2['green'] = [0, 0]
            joint_positions2['green'][0] = joint_positions2['red'][0]
            joint_positions2['green'][1] = joint_positions1['green'][1]

        if (joint_positions1['red'] is None):
            print("joint_positions1 occluded")
            joint_positions1[''] = [0, 0]
            joint_positions1['red'][0] = joint_positions1['green'][0]
            joint_positions1['red'][1] = joint_positions2['red'][1]

        if (joint_positions2['red'] is None):
            print("joint_positions2 occluded")
            joint_positions2['red'] = [0, 0]
            joint_positions2['red'][0] = joint_positions2['green'][0]
            joint_positions2['red'][1] = joint_positions1['red'][1]

        return joint_positions1, joint_positions2

    def compute_joint_position(self, joint_positions1, joint_positions2):
        pos_x = joint_positions2[0]
        pos_y = joint_positions1[0]
        pos_z = (joint_positions1[1] + joint_positions2[1]) / 2
        return np.array([pos_x, pos_y, pos_z])

    def transform_to_relative_pos(self, pos, origin):
        return np.array([
            pos[0] - origin[0],
            pos[1] - origin[1],
            origin[2] - pos[2]])

    def compute_distance_between_joints(self, joint1_pos, joint2_pos):
        return norm(joint1_pos - joint2_pos)

    def calculate_angles(self):
        # Uses least squares to compute joint angles.
        # The residuals function uses the green and red positions computed using opencv and
        # compares it to the forward kinematics.
        # Since we have 4 angles we need 4 differences, we are using green_x, green_y, green_z and red_x

        def f(transient_angles):

            t1 = self.kinematics.transform(self.yellow_blue_distance_meters, transient_angles[0] - np.pi / 2, 0, -np.pi / 2)
            t2 = self.kinematics.transform(0, transient_angles[1] - np.pi / 2, 0, np.pi / 2)
            t3 = self.kinematics.transform(0, transient_angles[2], self.blue_green_distance_meters, -np.pi / 2)
            t4 = self.kinematics.transform(0, transient_angles[3], self.green_red_distance_meters, 0)

            green_kinematics = t1.dot(t2).dot(t3)  # Kinematics up to green joint
            green_kinematics_x = green_kinematics[0, 3]
            green_kinematics_y = green_kinematics[1, 3]
            green_kinematics_z = green_kinematics[2, 3]

            red_kinematics = green_kinematics.dot(t4)  # Kinematics up to red joint
            red_kinematics_x = red_kinematics[0, 3]
            red_kinematics_y = red_kinematics[1, 3]
            red_kinematics_z = red_kinematics[2, 3]

            return np.array([
                self.green_pos[0] * self.meter_to_pixel_factor - green_kinematics_x,
                self.green_pos[1] * self.meter_to_pixel_factor - green_kinematics_y,
                self.green_pos[2] * self.meter_to_pixel_factor - green_kinematics_z,
                self.red_pos[0] * self.meter_to_pixel_factor - red_kinematics_x,
                self.red_pos[1] * self.meter_to_pixel_factor - red_kinematics_y,
                self.red_pos[2] * self.meter_to_pixel_factor - red_kinematics_z
            ])

        bounds = (np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2]),
                  np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]))

        angles = least_squares(f, self.initial_angles, jac="3-point", loss="cauchy", bounds=bounds)

        self.theta1 = angles.x[0]
        self.theta2 = angles.x[1]
        self.theta3 = angles.x[2]
        self.theta4 = angles.x[3]

        return angles.x

    def estimated_angles(self):
        return np.array([self.theta1, self.theta2, self.theta3, self.theta4])

    def compute_target_position(self, debug):
        target_position1 = self.common_image1.compute_target_center()
        target_position2 = self.common_image2.compute_target_center()

        if (target_position1 is None or target_position2 is None):
            return None

        self.target_abs_pos = self.compute_joint_position(target_position1, target_position2)
        self.target_pos = self.transform_to_relative_pos(self.target_abs_pos, self.origin_pos)

        if (debug):
            print("\nTarget position")
            print(self.target_pos)
            # self.common_image1.compute_blob_center(self.common_image1.lower_target, self.common_image1.upper_target, debug=True)
            # self.common_image2.compute_blob_center(self.common_image2.lower_target, self.common_image2.upper_target,
            #                                        debug=True)

            cv2.circle(self.cv_image1,
                       (int(self.target_abs_pos[1]), int(self.target_abs_pos[2])), 5,
                       (90, 10, 205), thickness=2, lineType=8)
            cv2.circle(self.cv_image2,
                       (int(self.target_abs_pos[0]), int(self.target_abs_pos[2])), 5,
                       (90, 10, 205), thickness=2, lineType=8)
            # cv2.imshow('Target Y-Z', self.cv_image1)
            # cv2.imshow('Target X-Z', self.cv_image2)
            # cv2.waitKey(1)

        # Target position correction
        return self.target_pos * np.array([1.07, 0.841, 1.07]) + np.array([0, 0, 13])

    def control_closed(self, debug=False):
        # P gain
        K_p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 8]])
        # D gain
        K_d = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step

        # if (debug):
        #     print("\ncontrol_closed")
        #     print("dt")
        #     print(dt)

        # if (dt<0.1):
        #    return
        #print(dt)

        self.time_previous_step = cur_time

        # robot end-effector position
        red_pos = self.red_pos * self.meter_to_pixel_factor

        # desired position
        target_pos_d = self.target_pos * self.meter_to_pixel_factor

        # estimate derivative of error
        self.error_d = ((target_pos_d - red_pos) - self.error) / dt

        print(self.error)

        # estimate error
        self.error = target_pos_d - red_pos

        jacobian = self.kinematics.jacobian(self)

        J_inv = np.linalg.pinv(jacobian)  # calculating the pseudo inverse of Jacobian

        dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) +
                              np.dot(K_p, self.error.transpose())))  # control input (angular velocity of joints)

        # if (debug):
        #     print(jacobian)
        #     print(dq_d)
        #     print(self.error)

        q_d = self.estimated_angles() + (dt * dq_d)  # control input (angular position of joints)

        return q_d