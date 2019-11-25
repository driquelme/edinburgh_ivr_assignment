import numpy as np
from math import cos, sin


class Kinematics:

    def kinematics(self, joint_state):
        t1 = self.transform(joint_state.yellow_blue_distance_meters, joint_state.theta1 - np.pi / 2, 0, -np.pi / 2)
        t2 = self.transform(0, joint_state.theta2 - np.pi / 2, 0, np.pi / 2)
        t3 = self.transform(0, joint_state.theta3, joint_state.blue_green_distance_meters, -np.pi / 2)
        t4 = self.transform(0, joint_state.theta4, joint_state.green_red_distance_meters, 0)
        return t1.dot(t2).dot(t3).dot(t4)

    def transform(self, d, theta, a, alpha):
        return np.array([
            [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
            [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def jacobian(self, joint_state):
        return np.array([
            [
                2 * cos(joint_state.theta1) * cos(joint_state.theta2) * sin(joint_state.theta4) +
                3* cos(joint_state.theta1) * cos(joint_state.theta3) * sin(joint_state.theta2) +
                cos(joint_state.theta4) * (
                        2* cos(joint_state.theta1) * cos(joint_state.theta3) * sin(joint_state.theta2) -
                        2* sin(joint_state.theta1) * sin(joint_state.theta3)) -
                3* sin(joint_state.theta1) * sin(joint_state.theta3),
                2* cos(joint_state.theta2) * cos(joint_state.theta3) * cos(joint_state.theta4) * sin(
                    joint_state.theta1) + 3* cos(joint_state.theta2) * cos(joint_state.theta3) * sin(
                    joint_state.theta1) - 2* sin(joint_state.theta1) * sin(joint_state.theta2) * sin(
                    joint_state.theta4),
                3* cos(joint_state.theta1) * cos(joint_state.theta3) + cos(joint_state.theta4) * (
                        2* cos(joint_state.theta1) * cos(joint_state.theta3) - 2* sin(joint_state.theta1) * sin(
                    joint_state.theta2) * sin(joint_state.theta3)) - 3* sin(joint_state.theta1) * sin(
                    joint_state.theta2) * sin(joint_state.theta3),
                2* cos(joint_state.theta2) * cos(joint_state.theta4) * sin(joint_state.theta1) - sin(
                    joint_state.theta4) * (
                        2* cos(joint_state.theta1) * sin(joint_state.theta3) + 2* cos(joint_state.theta3) * sin(
                    joint_state.theta1) * sin(joint_state.theta2)),
            ],
            [
                3* cos(joint_state.theta1) * sin(joint_state.theta3) + 2* cos(joint_state.theta2) * sin(
                    joint_state.theta1) * sin(joint_state.theta4) + 3* cos(joint_state.theta3) * sin(
                    joint_state.theta1) * sin(joint_state.theta2) + cos(joint_state.theta4) * (
                        2* cos(joint_state.theta1) * sin(joint_state.theta3) + 2* cos(joint_state.theta3) * sin(
                    joint_state.theta1) * sin(joint_state.theta2)),
                -2* cos(joint_state.theta1) * cos(joint_state.theta2) * cos(joint_state.theta3) * cos(
                    joint_state.theta4) - 3* cos(joint_state.theta1) * cos(joint_state.theta2) * cos(
                    joint_state.theta3) + 2* cos(joint_state.theta1) * sin(joint_state.theta2) * sin(
                    joint_state.theta4),
                3* cos(joint_state.theta1) * sin(joint_state.theta2) * sin(joint_state.theta3) + 3* cos(
                    joint_state.theta3) * sin(joint_state.theta1) + cos(joint_state.theta4) * (
                        2* cos(joint_state.theta1) * sin(joint_state.theta2) * sin(joint_state.theta3) + 2* cos(
                    joint_state.theta3) * sin(joint_state.theta1)),
                -2* cos(joint_state.theta1) * cos(joint_state.theta2) * cos(joint_state.theta4) - sin(
                    joint_state.theta4) * (
                        -2* cos(joint_state.theta1) * cos(joint_state.theta3) * sin(joint_state.theta2) + 2* sin(
                    joint_state.theta1) * sin(joint_state.theta3)),
            ],
            [
                0,
                -2* cos(joint_state.theta2) * sin(joint_state.theta4) - 2* cos(joint_state.theta3) * cos(
                    joint_state.theta4) * sin(joint_state.theta2) - 3* cos(joint_state.theta3) * sin(
                    joint_state.theta2),
                -2* cos(joint_state.theta2) * cos(joint_state.theta4) * sin(joint_state.theta3) - 3* cos(
                    joint_state.theta2) * sin(joint_state.theta3),
                -2* cos(joint_state.theta2) * cos(joint_state.theta3) * sin(joint_state.theta4) - 2* cos(
                    joint_state.theta4) * sin(joint_state.theta2),
            ],
        ])
