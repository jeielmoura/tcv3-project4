DATA_FOLDER = './data'
TEST_FOLDER = DATA_FOLDER + '/' + 'test'   	# folder with testing images
TRAIN_FOLDER = DATA_FOLDER + '/' + 'train' 	# folder with training images

MODEL_FOLDER = './model'

RESULT_FOLDER = './result'

SEMISUPERVISED_FILE = 'semi-supervised.txt'

IMAGE_HEIGHT = 77  # height of the image
IMAGE_WIDTH = 71   # width of the image
NUM_CHANNELS = 1   # number of channels of the image
NUM_IMAGES = -1    # number of train images

TRANSLATE_LIMIT = 3

#ensemble = 20%
#train = 70%
#valid = 10% + ensemble

ENSEMBLE_SPLIT_RATE = 0.8	# split rate for training and validation sets
TRAIN_SPLIT_RATE = 0.875	# split rate for training and validation sets

ENSEMBLE_SEED = 42

LEARNING_RATES = [0.001, 0.00055, 0.0001]
LEARNING_RATE_DECAY = 0.95

NUM_EPOCHS_LIMIT = 20

BATCH_SIZE = 64