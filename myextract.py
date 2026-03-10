import numpy as np

def calculate_min_intensity(image):
    return np.min(image)

def calculate_max_intensity(image):
    return np.max(image)

def calculate_mean_intensity(image):
    return np.mean(image)

def calculate_average_intensity(image):
    mean_intensity = calculate_mean_intensity(image)
    return mean_intensity / 255.0

def calculate_standard_deviation(image):
    return np.std(image)

def calculate_spectral_entropy(image):
    hist = np.histogram(image, bins=256, range=(0, 255))[0]
    hist = hist / (image.size + 1e-7)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy

def extract_features(image):
    min_intensity = calculate_min_intensity(image)

    max_intensity = calculate_max_intensity(image)

    mean_intensity = calculate_mean_intensity(image)

    average_intensity = calculate_average_intensity(image)

    standard_deviation = calculate_standard_deviation(image)

    spectral_entropy = calculate_spectral_entropy(image)

    return [min_intensity, max_intensity, mean_intensity, average_intensity, standard_deviation, spectral_entropy]

