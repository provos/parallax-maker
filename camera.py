# (c) 2024 Niels Provos

import numpy as np


class Camera:
    __slots__ = ('_camera_position', '_camera_distance',
                 '_max_distance', '_focal_length')

    def __init__(self, distance=100, max_distance=500, focal_length=100):
        """
        Initializes a Camera object.

        It isn't really a camera but encapsulates related properties.

        Args:
            distance (float): The distance of the camera from the scene.
            max_distance (float): The maximum distance/depth of the scene.
            focal_length (float): The focal length of the camera.

        Returns:
            None
        """
        self.camera_position = np.array([0, 0, -distance], dtype=np.float32)
        self.camera_distance = distance
        self.max_distance = max_distance
        self.focal_length = focal_length

    def setup_camera_and_cards(self, image_slices, depths, sensor_width=35.0):
        """
        Set up the camera matrix and the card corners in 3D space.

        Args:
            image_slices (list): A list of image slices.
            depths (list): A list of threshold depths for each image slice.
            sensor_width (float, optional): The width of the camera sensor. Defaults to 35.0.

        Returns:
            tuple: A tuple containing the camera matrix and a list of card corners in 3D space.
        """
        num_slices = len(image_slices)
        image_height, image_width, _ = image_slices[0].shape

        # Calculate the focal length in pixels
        focal_length_px = (image_width * self.focal_length) / sensor_width

        # Set up the camera intrinsic parameters
        camera_matrix = np.array([[focal_length_px, 0, image_width / 2],
                                [0, focal_length_px, image_height / 2],
                                [0, 0, 1]], dtype=np.float32)

        # Set up the card corners in 3D space
        card_corners_3d_list = []
        # The thresholds start with 0 and end with 255. We want the closest card to be at 0.
        for i in range(num_slices):
            z = self.max_distance * ((255 - depths[i]) / 255.0)

            # Calculate the 3D points of the card corners
            card_width = (image_width * (z + self.camera_distance)) / focal_length_px
            card_height = (image_height * (z + self.camera_distance)) / focal_length_px

            card_corners_3d = np.array([
                [-card_width / 2, -card_height / 2, z],
                [card_width / 2, -card_height / 2, z],
                [card_width / 2, card_height / 2, z],
                [-card_width / 2, card_height / 2, z]
            ], dtype=np.float32)
            card_corners_3d_list.append(card_corners_3d)

        return camera_matrix, card_corners_3d_list


    def to_json(self):
        return {
            'position': self._camera_position.tolist(),
            'camera_distance': self._camera_distance,
            'max_distance': self._max_distance,
            'focal_length': self._focal_length
        }

    @staticmethod
    def from_json(data):
        camera = Camera()
        if 'position' in data:
            camera.camera_position = np.array(
                data['position'], dtype=np.float32)
        if 'camera_distance' in data:
            camera.camera_distance = data['camera_distance']
        if 'max_distance' in data:
            camera.max_distance = data['max_distance']
        if 'focal_length' in data:
            camera.focal_length = data['focal_length']

        return camera

    def __str__(self):
        return f"Camera(distance={self._camera_distance}, max_distance={self._max_distance}, " \
               f"focal_length={self._focal_length}, position={self._camera_position})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Camera):
            return False
        return (self.camera_distance == other.camera_distance and
                self.max_distance == other.max_distance and
                self.focal_length == other.focal_length)

    @property
    def camera_position(self):
        return self._camera_position

    @camera_position.setter
    def camera_position(self, value):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (3,):
            raise ValueError(
                "camera_position must be a numpy array of dtype 'float32' with shape (3,)")
        self._camera_position = value

    @property
    def camera_distance(self):
        return self._camera_distance

    @camera_distance.setter
    def camera_distance(self, value):
        if not isinstance(value, (float, int)) or value < 0:
            raise ValueError("camera_distance must be a non-negative number")
        self._camera_distance = value

    @property
    def max_distance(self):
        return self._max_distance

    @max_distance.setter
    def max_distance(self, value):
        if not isinstance(value, (float, int)) or value < 0:
            raise ValueError("max_distance must be a non-negative number")
        self._max_distance = value

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        if not isinstance(value, (float, int)) or value <= 0:
            raise ValueError("focal_length must be a positive number")
        self._focal_length = value
