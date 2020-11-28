# Seam Carving algorithm based on 4gn3s work (https://github.com/4gn3s/seam-carving).
# Local Binary Pattern (LBP) algorithm based on the work of timvandermeij (https://github.com/timvandermeij/lbp.py).

# Algorithm Developed to carry out the Course Conclusion Work (Trabalho de Conclusão de Curso - TCC) in Computer Science 
# at Universidade Estadual Paulista Júlio de Mesquita Filho Campus of Bauru (UNESP)
# Work developed by Gabriel Vieira under the guidance of Prof. Dr. Kelton Costa  
#
# Algorithm responsible for creating the database that feeds a CNN Neural Network
# Application of the Seam Carving and LBP technique at different rates: [3%, 6%, 9%, 12%, 15%, 18%, 21%, 30%, 40%, 50%] 
# rates = ['three_percent', 'six_percent', 'nine_percent', 'twelve_percent', 
#     'fifteen_percent', 'eighteen_percent', 'twenty_one_percent', 'thirty_percent', 'forty_percent', 'fifty_percent']

#---IMPORTS
from skimage import data, transform, util, filters, color
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# ---GLOBAL VARIABLES
current_image = 10001
current_image_max = 10001 + 5150
img_path = '/home/user/Desktop/TCC_fazendo/dataset/dataset_uncompressed_images/{}.bmp' 
interations = 0

#---Settings the LBP
radius = 1 # radius size
n_points = 8 * radius # n neighboring points

#---Main of Project
while current_image < current_image_max:
    print('-----> LBP image: ' + str(current_image))

    #---Original Image ---
    img = mpimg.imread(img_path.format(current_image)) # Read an image from a file into an array
    # plt.title('Original Image')
    plt.axis('off')
    plt.imshow(img) # Display data as an image
    # plt.show()
    
    #---Performs LBP
    lbp_gray = color.rgb2gray(img)
    lbp = local_binary_pattern(lbp_gray,n_points,radius,'default')
    plt.figure()

    # plt.title('LBP of Seam carved image')
    plt.axis('off')
    plt.imshow(lbp)
    plt.savefig('/home/user/Desktop/TCC_fazendo/dataset/dataset_uncompressed_images_lbp/{}.png'.format(current_image)) # Create a new figure

    current_image += 1         