class Config():
    def __init__(self):
        self.debug = True
        self.input_size = 96 # Side length of square image patch
        self.batch_size = 2 # Batch size of patches Note: 11 gig gpu will max batch of 5
        self.k = 64 # Number of classes
        self.num_epochs = 250#250 for real
        self.data_dir = "./datasets/BSDS300/images/" # Directory of images
        self.showdata = False # Debug the data augmentation by showing the data we're training on.

        self.useInstanceNorm = True # Instance Normalization
        self.useBatchNorm = False # Only use one
        self.useDropout = True
        self.drop = 0.2

        # Each item in the following list specifies a module.
        # Each item is the number of input channels to the module.
        # The number of output channels is 2x in the encoder, x/2 in the decoder.
        self.encoderLayerSizes = [64, 128, 256]
        self.decoderLayerSizes = [512, 256]

        self.showSegmentationProgress = True
        self.segmentationProgressDir = './latent_images/'

        self.variationalTranslation = 0 # Pixels, 0 for off. 1 works fine
