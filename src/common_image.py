
import cv2
import numpy as np

class Common_image:

    def __init__(self, image):
        self.image = image
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        self.lower_yellow = np.array([30, 50, 50])
        self.upper_yellow = np.array([50, 255, 255])

        self.lower_blue = np.array([100, 50, 50])
        self.upper_blue = np.array([130, 255, 255])

        self.lower_green = np.array([50, 50, 50])
        self.upper_green = np.array([100, 255, 255])

        self.lower_red = np.array([0, 200, 100])
        self.upper_red = np.array([20, 255, 255])

        self.lower_target = np.array([15, 50, 10])
        self.upper_target = np.array([25, 255, 255])

    def compute_blob_contours(self, lower_bound, upper_bound, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, debug=False):
        mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
        mask2 = cv2.erode(mask, None, iterations=2)
        mask2 = cv2.dilate(mask2, None, iterations=2)

        if (debug):
            cv2.imshow('compute_blob_contours', mask)
            cv2.waitKey(1)

        return cv2.findContours(mask2.copy(), mode, method)[-2]

    def moments_coords(self, contour):
        M = cv2.moments(contour)
        return np.array([
            int(M["m10"] / M["m00"]),
            int(M["m01"] / M["m00"])])

    def compute_blob_center(self, lower_bound, upper_bound, debug=False):
        contours = self.compute_blob_contours(lower_bound, upper_bound)
        contour = max(contours, key=cv2.contourArea)

        if (debug):
            print(contour)

        return self.moments_coords(contour)

    def compute_meter_to_pixel_factor(self, coords1, coords2, distance):
        dist = np.sum((coords1 - coords2) ** 2)
        return distance / np.sqrt(dist)

    def compute_angle(self, coords1, coords2):
        return np.arctan2(coords1[0] - coords2[0], coords1[1] - coords2[1])

    def compute_joint_positions(self):
        coord_yellow = self.compute_blob_center(self.lower_yellow, self.upper_yellow)
        coord_blue = self.compute_blob_center(self.lower_blue, self.upper_blue)
        coord_green = self.compute_blob_center(self.lower_green, self.upper_green)
        coord_red = self.compute_blob_center(self.lower_red, self.upper_red)
        return {
            'yellow': coord_yellow,
            'blue': coord_blue,
            'green': coord_green,
            'red': coord_red
        }

    def compute_arm_angles(self):
        coord_yellow = self.compute_blob_center(self.lower_yellow, self.upper_yellow)
        coord_blue = self.compute_blob_center(self.lower_blue, self.upper_blue)
        coord_green = self.compute_blob_center(self.lower_green, self.upper_green)
        coord_red = self.compute_blob_center(self.lower_red, self.upper_red)
        meter_to_pixel_factor = self.compute_meter_to_pixel_factor(coord_blue, coord_yellow, 2)

        joint1_angle = self.compute_angle(
            meter_to_pixel_factor * coord_yellow,
            meter_to_pixel_factor * coord_blue)

        joint2_angle = self.compute_angle(
            meter_to_pixel_factor * coord_blue,
            meter_to_pixel_factor * coord_green) - joint1_angle

        joint3_angle = self.compute_angle(
            meter_to_pixel_factor * coord_green,
            meter_to_pixel_factor * coord_red) - joint2_angle - joint1_angle

        return np.array([joint1_angle, joint2_angle, joint3_angle])

    # Ref: https://pysource.com/2018/09/25/simple-shape-detection-opencv-with-python-3/
    def compute_target_center(self, debug=False):

        # Precompute meter to pixel factor
        coord_yellow = self.compute_blob_center(self.lower_yellow, self.upper_yellow)
        coord_blue = self.compute_blob_center(self.lower_blue, self.upper_blue)
        meter_to_pixel_factor = self.compute_meter_to_pixel_factor(coord_blue, coord_yellow, 2)

        # Detect and cycle over contours until we detect a circle shape
        contours = self.compute_blob_contours(self.lower_target, self.upper_target, cv2.RETR_LIST, cv2.RETR_CCOMP)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            if (debug):
                print(approx)

            if (len(approx) > 6): # Circle shape
                if (debug):
                    cv2.drawContours(self.hsv_image, [approx], 0, (0), 1)
                    cv2.imshow('window2', self.hsv_image)
                    cv2.waitKey(1)
                    print(len(approx))

                # Compute target position relative to the arm base
                target_pos = self.moments_coords(contour)
                target_rel_pos = np.absolute(target_pos - coord_yellow) * meter_to_pixel_factor

                if (debug):
                    print(target_rel_pos)

                return target_rel_pos