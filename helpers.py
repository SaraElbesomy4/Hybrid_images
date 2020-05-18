# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
from scipy.signal import gaussian, convolve2d, correlate2d



def my_imfilter(image: np.ndarray, filter: np.ndarray):

  """
  Your function should meet the requirements laid out on the project webpage.
  Apply a filter to an image. Return the filtered image.
  Inputs:
  - image -> numpy nd-array of dim (m, n, c) for RGB images or numpy nd-array of dim (m, n) for gray scale images
  - filter -> numpy nd-array of odd dim (k, l)
  Returns
  - filtered_image -> numpy nd-array of dim (m, n, c) or numpy nd-array of dim (m, n)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
  """

  # filter demnsions:
  filter_rows = filter.shape[0]
  filter_columns = filter.shape[1]

  assert (((filter_rows % 2) or (filter_columns % 2)) != 0), " Size of the kernel should be odd "


  img_shape = image.shape

  if len(img_shape) == 3 :

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

  # image demnsions:
    image_rows = r.shape[0]
    image_columns = r.shape[1]



  # Padding the image:
    pad_rows = (filter_rows-1) // 2
    pad_columns = (filter_columns-1) // 2
    padded_img = []
    padded_img.append(np.pad(r, ((pad_rows, pad_rows), (pad_columns, pad_columns)), 'constant', constant_values=0))
    padded_img.append(np.pad(g, ((pad_rows, pad_rows), (pad_columns, pad_columns)), 'constant', constant_values=0))
    padded_img.append(np.pad(b, ((pad_rows, pad_rows), (pad_columns, pad_columns)), 'constant', constant_values=0))

    filtered_image = np.zeros_like(r)

    for channel in padded_img:
      new_img = []
      for m in range(image_rows):
        for n in range(image_columns):
          summ = 0
          summ = np.sum(np.multiply(channel[m:m+filter_rows, n:n+filter_columns], filter))
          new_img.append(summ)
      new_img = np.asarray(new_img)
      new_img = new_img.reshape(image_rows, image_columns)
      filtered_image = np.dstack((filtered_image, new_img))

    filtered_image = filtered_image[:, :, 1:]

  elif len(img_shape) == 2:
    # image demnsions:
    image_rows = image.shape[0]
    image_columns = image.shape[1]

    # Padding the image:
    pad_rows = (filter_rows - 1) // 2
    pad_columns = (filter_columns - 1) // 2

    padded_img = []
    padded_img.append(np.pad(image , ((pad_rows, pad_rows), (pad_columns, pad_columns)), 'constant', constant_values=0))

    padded_img = np.asarray(padded_img)
    padded_img = padded_img.reshape(image_rows+ (2*pad_rows), image_columns+(2*pad_columns))

    filtered_image = np.zeros_like(image)


    new_img = []
    for m in range(image_rows):
      for n in range(image_columns):
        summ = 0
        summ = np.sum(np.multiply(padded_img[m:m + filter_rows, n:n + filter_columns], filter))
        new_img.append(summ)
    new_img = np.asarray(new_img)
    new_img = new_img.reshape(image_rows, image_columns)


  return filtered_image


# the same previous function but with reflection padding instead of zeros

def my_imfilter_reflect(image: np.ndarray, filter: np.ndarray):
  # filter demnsions:
  filter_rows = filter.shape[0]
  filter_columns = filter.shape[1]

  assert (((filter_rows % 2) or (filter_columns % 2)) != 0), " Size of the kernel should be odd "

  img_shape = image.shape

  if len(img_shape) == 3:

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # image demnsions:
    image_rows = r.shape[0]
    image_columns = r.shape[1]

    # Padding the image:
    pad_rows = (filter_rows - 1) // 2
    pad_columns = (filter_columns - 1) // 2
    padded_img = []
    padded_img.append(np.pad(r, ((pad_rows, pad_rows), (pad_columns, pad_columns)), 'reflect'))
    padded_img.append(np.pad(g, ((pad_rows, pad_rows), (pad_columns, pad_columns)), 'reflect'))
    padded_img.append(np.pad(b, ((pad_rows, pad_rows), (pad_columns, pad_columns)), 'reflect'))

    filtered_image = np.zeros_like(r)

    for channel in padded_img:
      new_img = []
      for m in range(image_rows):
        for n in range(image_columns):
          summ = 0
          summ = np.sum(np.multiply(channel[m:m + filter_rows, n:n + filter_columns], filter))
          new_img.append(summ)
      new_img = np.asarray(new_img)
      new_img = new_img.reshape(image_rows, image_columns)
      filtered_image = np.dstack((filtered_image, new_img))

    filtered_image = filtered_image[:, :, 1:]

  elif len(img_shape) == 2:
    # image demnsions:
    image_rows = image.shape[0]
    image_columns = image.shape[1]

    # Padding the image:
    pad_rows = (filter_rows - 1) // 2
    pad_columns = (filter_columns - 1) // 2

    padded_img = []
    padded_img.append(np.pad(image, ((pad_rows, pad_rows), (pad_columns, pad_columns)), 'reflect'))

    padded_img = np.asarray(padded_img)
    padded_img = padded_img.reshape(image_rows + (2 * pad_rows), image_columns + (2 * pad_columns))

    filtered_image = np.zeros_like(image)

    new_img = []
    for m in range(image_rows):
      for n in range(image_columns):
        summ = 0
        summ = np.sum(np.multiply(padded_img[m:m + filter_rows, n:n + filter_columns], filter))
        new_img.append(summ)
    new_img = np.asarray(new_img)
    new_img = new_img.reshape(image_rows, image_columns)


  return filtered_image


def create_gaussian_filter(ksize: int, sigma: float):
  x = gaussian(M=ksize, std=sigma)
  y = gaussian(M=ksize, std=sigma)
  x_1, y_1 = np.meshgrid(x, y)
  gaussian_kernel = x_1 * y_1
  gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
  return gaussian_kernel


def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
  """
 Inputs:
 - image1 -> The image from which to take the low frequencies.
 - image2 -> The image from which to take the high frequencies.
 - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                       blur that will remove high frequencies.

 Task:
 - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
 - Combine them to create 'hybrid_image'.
"""

  assert image1.shape == image2.shape

  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
  # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel
  kernel_size = 3 * cutoff_frequency
  kernel = create_gaussian_filter(kernel_size, cutoff_frequency)
  # low_frequencies = np.zeros(image1.shape,dtype = float)
  # low_frequencies[:,:,0] = correlate2d(image1[:,:,0], kernel, 'same') ###should be changed
  # low_frequencies[:,:,1] = correlate2d(image1[:,:,1], kernel, 'same') ###should be changed
  # low_frequencies[:,:,2] = correlate2d(image1[:,:,2], kernel, 'same') ###should be changed
  low_frequencies = my_imfilter(image1, kernel)
  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  # high_frequencies = np.zeros(image1.shape,dtype = float)
  # high_frequencies[:,:,0] = image2[:,:,0] - correlate2d(image2[:,:,0], kernel, 'same') ###should be changed
  # high_frequencies[:,:,1] = image2[:,:,1] - correlate2d(image2[:,:,1], kernel, 'same') ###should be changed
  # high_frequencies[:,:,2] = image2[:,:,2] - correlate2d(image2[:,:,2], kernel, 'same') ###should be changed
  high_frequencies = image2 - my_imfilter(image2, kernel)
  # (3) Combine the high frequencies and low frequencies
  hybrid_image = low_frequencies + high_frequencies

  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to
  # gen_hybrid_image().
  # One option is to clip (also called clamp) all values below 0.0 to 0.0,
  # and all values larger than 1.0 to 1.0.
  # high_frequencies[high_frequencies< 0.0] = 0.0
  # high_frequencies[high_frequencies> 1.0] = 1.0
  hybrid_image[hybrid_image < 0.0] = 0.0
  hybrid_image[hybrid_image > 1.0] = 1.0
  # (5) As a good software development practice you may add some checks (assertions) for the shapes
  # and ranges of your results. This can be performed as test for the code during development or even
  # at production!

  return low_frequencies, high_frequencies, hybrid_image


# In[4]:


def vis_hybrid_image(hybrid_image: np.ndarray):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect')
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
