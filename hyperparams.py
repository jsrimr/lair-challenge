from data import label_encoder, csv_feature_dict


SEED = 42
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 256
CLASS_N = len(label_encoder)
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 512
NUM_FEATURES = len(csv_feature_dict)
MAX_LEN = 24*6
DROPOUT_RATE = 0.1
EPOCHS = 10
NUM_WORKERS = 16
