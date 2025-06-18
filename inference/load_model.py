import utility.config as config

# This module initializes a Detectron2 predictor with the specified configuration.
# It ensures that the predictor is created only once and can be reused across multiple calls.
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2 import model_zoo

            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(config.BASE_CONFIG_PATH))
            cfg.MODEL.WEIGHTS = config.MODEL_PATH
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.SCORE_THRESHOLD
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.NMS_THRESHOLD
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.NUM_CLASSES
            cfg.MODEL.DEVICE = config.device

            predictor = DefaultPredictor(cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize predictor: {str(e)}")
    return predictor