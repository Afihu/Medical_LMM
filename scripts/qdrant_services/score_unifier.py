import math

# More advanced method, prioritizes both modalities, severely penalizes the model if one modality has low score
def geometric_mean(score_image, score_text):
    return math.sqrt(score_image * score_text)

# Basic method, change weights depending on which modality to prioritize
def weighted_average(score_image, score_text, weight_img, weight_txt):
    return ((weight_img * score_image) + weight_txt * score_text)
