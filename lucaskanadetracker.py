

import cv2
import numpy as np

class LucasKanadeTracker:
    _maximum_distance_same_object = 10     # pixels
    def __init__(self,image):
        # Creates a tracker that imitates Sort tracker in format for later comparison
        self._current_objects = []
        # In the form: [{'id':0, box': [84, 39, 99, 117], 'tracking': [[[84, 39]],[[99, 117]]], 'confidence': 0.9},etc]
        self._last_object_id = 0
        self._time_out_existence = 8                # frames
        
        self.lucas_kanade_parameters = dict(    winSize  = (15,15),
                                                maxLevel = 7,
                                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.old_image = image

    def update(self,detections, image = None):
        """
        Inputs a list in the form [{'box': [84, 39, 99, 117], 'confidence': 0.992, 'keypoints': {'mid_point': (104, 83), 'plate': (148, 75)}}]
        Returns a box and an object ID in the form [[x,y,w,h,ID], ... ,[x,y,w,h,ID]]
        """
        if self._current_objects == []:
            # If currently there are no objects, we just check to the confidence
            for detection in detections:
                if detection['confidence']>=0.5:
                    # We update the object with high confidence to match the standards required
                    self.add_new_object(detection)
        else:
            # If there are points we check if a moving poing matches the position of a new one
            for current_object in self._current_objects:
                old_position = LucasKanadeTracker.convert_to_numpy_lk(current_object['tracking'])
                new_position, _, _ = cv2.calcOpticalFlowPyrLK(self.old_image, image, old_position, None, **self.lucas_kanade_parameters)
                for detection in detections:
                    points_to_check = LucasKanadeTracker.convert_to_numpy_lk(LucasKanadeTracker.get_tracking_points(detection['box']))
                    if LucasKanadeTracker.compare_np_lk_points(new_position,points_to_check):
                        # If points are closer they are assigned no matter the confidence
                        current_life_span = current_object['life_span'] + 1
                        update_dict = { 'tracking': LucasKanadeTracker.get_tracking_points(detection['box']),
                                        'life_span': current_life_span}
                        # If a matching boxs is found the life_span remains constant
                        current_object.update(update_dict)
                    else:
                        # If not, but the confidence is great enough, we create a nuw object
                        self.add_new_object(detection)
                # The life span degreases every frame
                current_object['life_span'] = current_object['life_span'] - 1
        
        self.old_image = image
        return self._current_objects

    def add_new_object(self,detection):
        update_dict = { 'id': self._last_object_id,
                        'tracking': LucasKanadeTracker.get_tracking_points(detection['box']),
                        'life_span': 5}
        self._last_object_id += 1
        del detection['keypoints']
        detection.update(update_dict)
        self._current_objects.append(detection)

    @staticmethod
    def get_tracking_points(box_rectangle):
        x1, y1, x2, y2 = box_rectangle
        return [((x1+x2)//2,(y1+y2)//2),]
    
    @staticmethod
    def convert_to_numpy_lk(point_arrays):
        mi_array = []
        for point in point_arrays:
            mi_array.append([point])
        return np.array(mi_array)

    @staticmethod
    def convert_from_numpy_lk(np_array):
        mi_array = []
        for point in np_array:
            mi_array.append(point[0])
        return mi_array

    @staticmethod
    def compare_np_lk_points(points_1,points_2):
        # This method currently supports one point comparison only.
        # Returns True if the poinst are closer than the maximum distance:
        differenceArray = points_1[0][0] - points_2[0][0]
        if LucasKanadeTracker.squared_size_vector(differenceArray) < LucasKanadeTracker._maximum_distance_same_object**2:
            return True 
        else:
            return False

    @staticmethod
    def squared_size_vector(np_vector):
        return np_vector[0]**2+np_vector[1]**2
