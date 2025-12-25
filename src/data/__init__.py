from src.data.plantseg import (
    DISEASE_CLASSES,
    PlantSegDataset,
    PlantSegMulticlassDataset,
)
from src.data.plantvillage_mappings import (
    CLASSIFICATION_CLASSES,
    DISEASE_TO_CLASS_IDX,
    EXCLUDED_FOLDERS,
    NUM_CLASSIFICATION_CLASSES,
    PLANTVILLAGE_FOLDER_TO_CLASS,
    PLANTVILLAGE_FOLDER_TO_PLANT,
    PLANTVILLAGE_ONLY_DISEASES,
)
from src.data.plantvillage import (
    PlantSegClassificationDataset,
    PlantVillageDataset,
    get_combined_class_weights,
    get_combined_sample_weights,
)
