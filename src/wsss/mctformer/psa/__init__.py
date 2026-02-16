"""PSA (Pixel-wise Semantic Affinity) post-processing for CAM refinement.

Converts raw CAMs into refined pseudo segmentation masks using:
1. Affinity network training on CRF-processed CAMs
2. Random-walk inference to propagate labels

To be ported from MCTformer/psa/ in phase1-psa.
"""
