"""
Config file with common constants
"""

# Path to data directory
data_directory = "../../data/faces/"

# Log file path
log_path = "/tmp/faces/log.html"

# Batch size to be used by prediction models
batch_size = 8

# Size of crops data generators should return
crop_size = 224

# Size of inputs models are trained on
image_shape = (crop_size, crop_size, 3)

# Step to be used to sample crops from images
step = 32

# Path to model file
model_path = "../../data/faces/models/model.h5"
