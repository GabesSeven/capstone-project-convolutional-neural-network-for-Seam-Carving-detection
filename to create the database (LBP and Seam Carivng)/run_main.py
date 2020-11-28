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

from PIL import Image
from seam_carving import SeamCarver

#---GLOBAL VARIAVLES
rates = ['three_percent', 'six_percent', 'nine_percent', 'twelve_percent', 'fifteen_percent', 
    'eighteen_percent', 'twenty_one_percent', 'thirty_percent', 'forty_percent', 'fifty_percent']
current_image = 10001
range_max_current_rate = current_image + 515  # 5150 / len(rates) 
img_path = '/home/user/Desktop/TCC_fazendo/dataset/dataset_uncompressed_images/{}.bmp' 
rates_num = [8, 16, 23, 31, 38, 46, 54, 77, 102, 128] # Considering pictures on size 256 x 256
interations = 0

#---Function Responsible for Seams transformation
def performs_seams():
    #---Original Image ---
    img = mpimg.imread(img_path.format(current_image))
    # plt.title('Original Image')
    plt.axis('off')
    plt.imshow(img) # Display data as an image
    # plt.show()
    sc = SeamCarver(img_path.format(current_image)) # Performs seam_carving.py
    
    #---Performs Seam Carving
    bigger = sc.resize((sc.image.width - int((rates_num[interations] / 2))), (sc.image.height - int((rates_num[interations] / 2))))
    plt.figure() # Create a new figure
    # plt.title('Seam Carving')
    plt.axis('off')
    plt.imshow(np.asarray(bigger.arrayObj()))
    
    #---Performs LBP
    performs_lbp(bigger._array, 'seam_carving')

    #---Performs Seam Insertion
    smaller = sc.resize((sc.image.width + int((rates_num[interations] / 2))), (sc.image.height + int((rates_num[interations] / 2))))
    plt.figure() 
    # plt.title('Seam Insertion')
    plt.axis('off')
    plt.imshow(np.asarray(smaller.arrayObj()))
    
    #---Performs LBP
    performs_lbp(smaller._array, 'seam_insertion')

    # plt.show()
    # plt.close('all')

#---Function Responsible for LBP transformation
def performs_lbp(sc, seam): 
    #---Settings the LBP
    radius = 1 # radius size
    n_points = 8 * radius # n neighboring points
    
    #---Performs LBP
    sc_gray = color.rgb2gray(sc)
    lbp = local_binary_pattern(sc_gray,n_points,radius,'default')
    plt.figure()
    # plt.title('LBP of Seam carved image')
    plt.axis('off')
    plt.imshow(lbp)
    plt.savefig('/home/user/Desktop/TCC_fazendo/dataset/images_differents_rates/{}/{}/{}.png'.format(seam, current_rate, current_image)) # Create a new figure


#---Main of Project
#---ALL RATES
for current_rate in rates:

    #---RATE 'three_percent' - 1.5% rows and 1.5% columns are removed
    #---RATE 'six_percent' - 3% rows and 3% columns are removed
    #---RATE 'nine_percent' - 4.5% rows and 4.5% columns are removed
    # so successively ...
    
    print('-----> Rate {}:'.format(current_rate))
    print('-----> Numbers of rows handled: {}'.format(rates_num[interations] / 2))
    print('-----> Numbers of columns handled: {}'.format(rates_num[interations] / 2))
    
    while current_image < range_max_current_rate:
        print('-----> Resizing image: ' + str(current_image))
        performs_seams()
        current_image += 1         

    range_max_current_rate += 515
    interations += 1