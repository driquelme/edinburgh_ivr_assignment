
import cv2
import numpy as np

class Common_image:

    def __init__(self, image):
        self.image = image
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        self.lower_yellow = np.array([30, 50, 50])
        self.upper_yellow = np.array([40, 255, 255])

        self.lower_blue = np.array([100, 50, 50])
        self.upper_blue = np.array([130, 255, 255])

        self.lower_green = np.array([50, 50, 50])
        self.upper_green = np.array([103, 255, 255])

        self.lower_red = np.array([0, 200, 100])
        self.upper_red = np.array([20, 255, 255])

        self.lower_target = np.array([15, 50, 10])
        self.upper_target = np.array([25, 255, 255])

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    #       Object Detection
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def compute_blob_contours(self, lower_bound, upper_bound, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, debug=False):
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
        mask2 = cv2.dilate(mask, kernel, iterations=3)

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
        if (len(contours)==0):
            return None
        contour = max(contours, key=cv2.contourArea)
        moments = self.moments_coords(contour)

        if (debug):
            import random
            print("\ncompute_blob_center_" + str(int(random.random()*100)))
            print(moments)
            cv2.circle(self.image, (moments[0], moments[1]), 12, (0, 255, 0), thickness=2, lineType=8)
            cv2.imshow('compute_blob_center', self.image)
            cv2.waitKey(1)

        return moments

    def compute_meter_to_pixel_factor(self, coords1, coords2, distance):
        dist = np.sum((coords1 - coords2) ** 2)
        return distance / np.sqrt(dist)

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


    # Ref: https://pysource.com/2018/09/25/simple-shape-detection-opencv-with-python-3/
    def compute_target_center(self, debug=False):

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

                if (debug):
                    print(target_pos)

                return target_pos