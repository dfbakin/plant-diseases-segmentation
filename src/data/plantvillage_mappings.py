"""
Class Index System:
- 0: healthy (all healthy samples)
- 1-115: PlantSeg diseases (from DISEASE_CLASSES[1:])
- 116-119: PlantVillage-only diseases (no PlantSeg ground truth)
"""
from src.data.plantseg import DISEASE_CLASSES, PlantSegDataset

# PlantVillage-only diseases
PLANTVILLAGE_ONLY_DISEASES = [
    "grape esca (black measles)",  # Grape___Esca_(Black_Measles)
    "peach bacterial spot",  # Peach___Bacterial_spot
    "tomato spider mites",  # Tomato___Spider_mites Two-spotted_spider_mite
    "tomato target spot",  # Tomato___Target_Spot
]

# Classification classes: healthy + PlantSeg diseases + PlantVillage-only diseases
# fmt: off
CLASSIFICATION_CLASSES = (
    ["healthy"]  # Index 0
    + DISEASE_CLASSES[1:]  # Indices 1-115
    + PLANTVILLAGE_ONLY_DISEASES  # Indices 116-119
)
# fmt: on

NUM_CLASSIFICATION_CLASSES = len(CLASSIFICATION_CLASSES)  # 120

DISEASE_TO_CLASS_IDX = {name: idx for idx, name in enumerate(CLASSIFICATION_CLASSES)}

# fmt: off
PLANTVILLAGE_FOLDER_TO_CLASS = {
    # Apple
    "Apple___Apple_scab": DISEASE_TO_CLASS_IDX["apple scab"],  # 4
    "Apple___Black_rot": DISEASE_TO_CLASS_IDX["apple black rot"],  # 1
    "Apple___Cedar_apple_rust": DISEASE_TO_CLASS_IDX["apple rust"],  # 3
    "Apple___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Blueberry
    "Blueberry___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Cherry
    "Cherry___Powdery_mildew": DISEASE_TO_CLASS_IDX["cherry powdery mildew"],  # 38
    "Cherry___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Corn
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": DISEASE_TO_CLASS_IDX["corn gray leaf spot"],  # 45
    "Corn___Common_rust": DISEASE_TO_CLASS_IDX["corn rust"],  # 47
    "Corn___Northern_Leaf_Blight": DISEASE_TO_CLASS_IDX["corn northern leaf blight"],  # 46
    "Corn___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Grape
    "Grape___Black_rot": DISEASE_TO_CLASS_IDX["grape black rot"],  # 59
    "Grape___Esca_(Black_Measles)": DISEASE_TO_CLASS_IDX["grape esca (black measles)"],  # 116
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": DISEASE_TO_CLASS_IDX["grape leaf spot"],  # 61
    "Grape___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Orange/Citrus
    "Orange___Haunglongbing_(Citrus_greening)": DISEASE_TO_CLASS_IDX["citrus greening disease"],  # 40
    # Peach
    "Peach___Bacterial_spot": DISEASE_TO_CLASS_IDX["peach bacterial spot"],  # 117
    "Peach___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Pepper
    "Pepper,_bell___Bacterial_spot": DISEASE_TO_CLASS_IDX["bell pepper bacterial spot"],  # 15
    "Pepper,_bell___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Potato
    "Potato___Early_blight": DISEASE_TO_CLASS_IDX["potato early blight"],  # 76
    "Potato___Late_blight": DISEASE_TO_CLASS_IDX["potato late blight"],  # 77
    "Potato___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Raspberry
    "Raspberry___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Soybean
    "Soybean___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Squash
    "Squash___Powdery_mildew": DISEASE_TO_CLASS_IDX["squash powdery mildew"],  # 90
    # Strawberry
    "Strawberry___Leaf_scorch": DISEASE_TO_CLASS_IDX["strawberry leaf scorch"],  # 92
    "Strawberry___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
    # Tomato
    "Tomato___Bacterial_spot": DISEASE_TO_CLASS_IDX["tomato bacterial leaf spot"],  # 97
    "Tomato___Early_blight": DISEASE_TO_CLASS_IDX["tomato early blight"],  # 98
    "Tomato___Late_blight": DISEASE_TO_CLASS_IDX["tomato late blight"],  # 99
    "Tomato___Leaf_Mold": DISEASE_TO_CLASS_IDX["tomato leaf mold"],  # 100
    "Tomato___Septoria_leaf_spot": DISEASE_TO_CLASS_IDX["tomato septoria leaf spot"],  # 102
    "Tomato___Spider_mites Two-spotted_spider_mite": DISEASE_TO_CLASS_IDX["tomato spider mites"],  # 118
    "Tomato___Target_Spot": DISEASE_TO_CLASS_IDX["tomato target spot"],  # 119
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": DISEASE_TO_CLASS_IDX["tomato yellow leaf curl virus"],  # 103
    "Tomato___Tomato_mosaic_virus": DISEASE_TO_CLASS_IDX["tomato mosaic virus"],  # 101
    "Tomato___healthy": DISEASE_TO_CLASS_IDX["healthy"],  # 0
}
# fmt: on

# Excluded folders (not plant/leaf images)
EXCLUDED_FOLDERS = {"Background_without_leaves"}

# Extract plant name from folder (e.g., "Apple___Black_rot" -> "Apple")
PLANTVILLAGE_FOLDER_TO_PLANT = {
    folder: folder.split("___")[0].replace(",_", " ").replace("_", " ")
    for folder in PLANTVILLAGE_FOLDER_TO_CLASS
}