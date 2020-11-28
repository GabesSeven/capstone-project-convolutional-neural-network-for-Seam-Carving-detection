import numpy as np
import scipy.misc
import scipy.ndimage
from PIL import Image as PILImage

# https://docs.python.org/2/tutorial/classes.html
# a name prefixed with an underscore mean “private” instance variables 
# “private” instance variables that cannot be accessed except from inside an object don’t exist in Python
class Image:
    def __init__(self, array=None, transposed=False):
        self._array = array
        self.greyscale_coeffs = [.299, .587, .144] # gray = .299*red + .587*green + .114*blue
        self.transposed = transposed
        self.greyscale_image = None
        self.sobel_image = None # sobel filter
        self.min_energy_image = None
    # https://python101.pythonlibrary.org/chapter25_decorators.html
    # Decorator in Python is a function that accepts another function as an argument
    # The decorator will usually modify or enhance the function it accepted and return the modified function
    
    #@property
    # This allows you to turn a class method into a class attribute
    # Convert class methods into read-only attributes
    # chamar uma função dentro de outra função, com decorator: attribute.func. Sem decorator: attribute.func() 
    @property
    def array(self):
        """
        :return: the image array (transposed if needed)
        """
        # https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
        # permute the dimensions of an array
        if self.transposed:
            if self.dim == 3:
                # (quantidade de novo(s) array(s), num de linha(s) novo array, num de coluna(s) novo array)
                return self._array.transpose(1, 0, 2)
            else:
                # (num de linha(s) novo array, num de coluna(s) novo array)
                return self._array.transpose(1, 0)
        return self._array

    @property
    def width(self):
        if self.transposed:
            return self._array.shape[0]
        return self._array.shape[1]

    @property
    def height(self):
        if self.transposed:
            return self._array.shape[1]
        return self._array.shape[0]

    @property
    def dim(self): # retorna a dimensão da imagem: (linha, coluna)
        if len(self._array.shape) > 2:
            return self._array.shape[2]
        return 2

    @classmethod
    def from_image(cls, image):
        return cls(image.array, image.transposed)

    # devolve o array da imagem original, verifica se a matriz de entrada não está trasnposta
    @classmethod
    def from_image_array(cls, array, transposed=False):
        return cls(array, transposed) # cls() é uma função contrutora que atualiza e retorna o array atualizado

    @classmethod
    def from_file(cls, image_file):
        return cls(scipy.misc.imread(image_file))

    @property
    def greyscale(self):
        """
        :return: greyscale image transposed if needed
        """
        if not self.greyscale_image:
            self.greyscale_image = np.dot(self.array[:, :, :3], self.greyscale_coeffs)
        return self.greyscale_image

    @property
    def energy(self):
        """
        Based on http://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy#1
        """
        # se não realizado o filtro sobel
        if not self.sobel_image:
            greyscale = self.greyscale.astype('int32') # converse o array '[.299, .587, .144]' np tipo 'int32'
            dx = scipy.ndimage.sobel(greyscale, 0)  # derivada horizontal, '0' significa axis='rows'
            dy = scipy.ndimage.sobel(greyscale, 1)  # derivada vertical, '1' significa axis='columns'
            self.sobel_image = np.hypot(dx, dy)  # This method returns Euclidean norm, sqrt(dx*dx + dy*dy), magnitude
            self.sobel_image *= 255.0 / np.max(self.sobel_image)  # normaliza o resultado, normalize
        return self.sobel_image
    
    @property
    def min_energy(self):
        """
        Converts energy values to cumulative energy values
        """
        # se não foi encontrado a mínima função de energia
        if not self.min_energy_image: 
            image = self.energy # 'energy()': encotra o matriz filtro Sobel, uma matriz modificada da matriz da imagem original
            self.min_energy_image = np.zeros((self.height, self.width)) # cria uma matriz composta só por '0'
            self.min_energy_image[0][:] = image[0][:] # primeira linha da matriz composta por '0' é a mesma da matriz filtro Sobel

            for i in range(self.height): # (0, self.height)
                for j in range(self.width): # (0, self.width)
                    if i == 0: # se primeira linha
                        self.min_energy_image[i, j] = image[i, j] # pixel da mínima energia = matriz filtro Sobel
                    elif j == 0: # se primeira coluna 
                        # pixel matriz mínima energia = matriz filtro Sobel +  menor pixel da coluna atual ou posterior  
                        self.min_energy_image[i, j] = image[i, j] + min(
                            self.min_energy_image[i - 1, j],
                            self.min_energy_image[i - 1, j + 1]
                        ) 
                    elif j == self.width - 1: # se última primeira coluna
                        # pixel matriz mínima energia = matriz filtro Sobel + menor pixel da coluna atual ou anterior
                        self.min_energy_image[i, j] = image[i, j] + min(
                            self.min_energy_image[i - 1, j - 1],
                            self.min_energy_image[i - 1, j]
                        )
                    else: # senão pode ser coluna atual, anterior ou posterior
                        self.min_energy_image[i, j] = image[i, j] + min(
                            self.min_energy_image[i - 1, j - 1],
                            self.min_energy_image[i - 1, j],
                            self.min_energy_image[i - 1, j + 1]
                        )
        return self.min_energy_image

    def debug(self, seam):
        """
        :param seam: current seam in the image (2 dim array with one column/row)
        :return: a debug image showing the actual image being processed with the currently chosen seam
        """
        image = self.array
        color = [255] * 3
        seam_array = seam.array
        size = seam.width if seam.width > seam.height else seam.height
        for i in range(size):
            if seam.transposed:
                image[i][seam_array[0][i]] = color
            else:
                image[i][seam_array[i][0]] = color
        return image

    def save(self, filename):
        im = PILImage.fromarray(self.array.astype('uint8'))
        im.save(filename)

    def show(self):
        im = PILImage.fromarray(self.array.astype('uint8'))
        im.show(self)

    def arrayObj(self):
        return PILImage.fromarray(self.array.astype('uint8'))

    # .fromarray(obj, mode=None): creates an image memory from an object exporting the array interface (using the buffer protocol).
    # .astype(): Copy of the array, cast to a specified type
    def PILresize(self, desired_width, desired_height):
        # https://pillow.readthedocs.io/en/4.2.x/reference/Image.html
        # The module also provides a number of factory functions, 
        # including functions to load images from files, and to create new images
        img = PILImage.fromarray(self.array.astype('uint8'))
        # função de 'seam_carving.py', responsável por modificar e retorna a imagem modificada
        return img.resize((desired_width, desired_height))
