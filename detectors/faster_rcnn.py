import tensorflow as tf 
from heads import RPNHead
from detectors import Detector


class FasterRCNN(Detector):
    def __init__(self, cfg, **kwargs):
        head = RPNHead(cfg)
        super().__init__(cfg, head=head, **kwargs)
    
    


