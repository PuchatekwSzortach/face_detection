"""
Config file with common constants
"""

# Path to data directory
data_directory = "../../data/faces/"

# Log file path
log_path = "/tmp/faces/log.html"

# Batch size to be used by prediction models
batch_size = 64

# Size of crops data generators should return
crop_size = 224

# Size of inputs models are trained on
image_shape = (crop_size, crop_size, 3)

# Stride to be used to sample crops from images
stride = 32


class FaceSearchConfiguration:
    """
    A simple class that bundles together common face search parameters
    """

    def __init__(self, crop_size, stride, batch_size):
        """
        Constructor
        :param crop_size: size of crops used to search for faces
        :param stride: stride between successive crops
        :param batch_size: batch size used by predictive model
        """

        self.crop_size = crop_size
        self.stride = stride
        self.batch_size = batch_size


face_search_config = FaceSearchConfiguration(crop_size=crop_size, stride=stride, batch_size=batch_size)

# Path to model file
model_path = "../../data/faces/models/model.h5"
