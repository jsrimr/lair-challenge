from utils import initialize
csv_feature_dict, label_encoder, label_decoder = initialize()

ROOT_DIR = 'data'
SEED = 42
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
BATCH_SIZE = 16
CLASS_N = len(label_encoder)
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 512
NUM_FEATURES = len(csv_feature_dict)
MAX_LEN = 24*6
DROPOUT_RATE = 0.1
EPOCHS = 10
NUM_WORKERS = 16
MODEL_NAME = "ConvNeXt-B-22k"
