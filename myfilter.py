import scipy

def gaussian_filter(input, sigma):
    output = scipy.ndimage.gaussian_filter(input, sigma)
    return output