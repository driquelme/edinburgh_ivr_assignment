
import numpy as np
import rospy

class Control:

    def __init__(self):
        # record the begining time
        self.time_trajectory = rospy.get_time()

    def forward_kinematics(self, joints):
        end_effector = np.array([3 * np.sin(joints[0]) +
                                 3 * np.sin(joints[0] + joints[1]) +
                                 3 * np.sin(joints.sum()),

                                 3 * np.cos(joints[0]) + 3 *
                                 np.cos(joints[0] + joints[1]) +
                                 3 * np.cos(joints.sum())])
        return end_effector

    def calculate_jacobian(self, joints):
        jacobian = np.array([[3 * np.cos(joints[0]) +
                              3 * np.cos(joints[0] + joints[1]) +
                              3 * np.cos(joints.sum()),

                              3 * np.cos(joints[0] + joints[1]) +
                              3 * np.cos(joints.sum()),
                              3 * np.cos(joints.sum())],

                             [-3 * np.sin(joints[0]) -
                              3 * np.sin(joints[0] + joints[1]) -
                              3 * np.sin(joints.sum()),

                              - 3 * np.sin(joints[0] + joints[1]) -
                              3 * np.sin(joints.sum()),
                              - 3 * np.sin(joints.sum())]])
        return jacobian

        # Define a circular trajectory

    def trajectory(self):
        # get current time
        cur_time = np.array([rospy.get_time() - self.time_trajectory])
        x_d = float(6 * np.cos(cur_time * np.pi / 100))
        y_d = float(6 + np.absolute(1.5 * np.sin(cur_time * np.pi / 100)))
        return np.array([x_d, y_d])

    def control_open(self, image):
        # estimate time step
        cur_time = rospy.get_time()
        dt = cur_time - self.time_previous_step2
        self.time_previous_step2 = cur_time
        q = self.detect_joint_angles(image)  # estimate initial value of joints'
        J_inv = np.linalg.pinv(self.calculate_jacobian(image))  # calculating the psudeo inverse of Jacobian
        # desired trajectory
        pos_d = self.trajectory()
        # estimate derivative of desired trajectory
        self.error_d = (pos_d - self.error) / dt
        self.error = pos_d
        q_d = q + (dt * np.dot(J_inv, self.error_d.transpose()))  # desired joint angles to follow the trajectory
        return q_d

    def control_closed(self, joints):
        # P gain
        K_p = np.array([[10, 0], [0, 10]])
        # D gain
        K_d = np.array([[0.1, 0], [0, 0.1]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time
        # robot end-effector position
        pos = self.detect_end_effector(joints)
        # desired trajectory
        pos_d = self.trajectory()
        # estimate derivative of error
        self.error_d = ((pos_d - pos) - self.error) / dt
        # estimate error
        self.error = pos_d - pos
        q = joints  # estimate initial value of joints'
        J_inv = np.linalg.pinv(self.calculate_jacobian(joints))  # calculating the psudeo inverse of Jacobian
        dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) +
                              np.dot(K_p, self.error.transpose())))  # control input (angular velocity of joints)
        q_d = q + (dt * dq_d)  # control input (angular position of joints)
        return q_d