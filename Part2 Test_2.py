#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
from skimage.util import img_as_float32
from skimage import io, img_as_ubyte
from skimage.transform import rescale
import matplotlib.pyplot as plt
from helpers import load_image, save_image, my_imfilter, gen_hybrid_image, vis_hybrid_image


image1 = load_image('../data/dog.bmp')
image2 = load_image('../data/cat.bmp')

# display the dog and cat images
plt.figure(figsize=(3,3))
plt.imshow((image1*255).astype(np.uint8))
plt.figure(figsize=(3,3))
plt.imshow((image2*255).astype(np.uint8))



## Hybrid Image Construction ##
# cutoff_frequency is the standard deviation, in pixels, of the Gaussian#
# blur that will remove high frequencies. You may tune this per image pair
# to achieve better results.
cutoff_frequency = 7
low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(image1, image2, cutoff_frequency)

## Visualize and save outputs ##
plt.figure()
plt.imshow((low_frequencies*255).astype(np.uint8))
plt.figure()
plt.imshow(((high_frequencies+0.5)*255).astype(np.uint8))
vis = vis_hybrid_image(hybrid_image)
plt.figure(figsize=(20, 20))
plt.imshow(vis)


#
#np.clip(high_frequencies, -1,1)
save_image('../results/low_frequencies.jpg', low_frequencies)
save_image('../results/high_frequencies.jpg', (high_frequencies - np.min(high_frequencies))/(np.max(high_frequencies)-np.min(high_frequencies)) )
save_image('../results/hybrid_image.jpg', hybrid_image)
save_image('../results/hybrid_image_scales.jpg', vis)


# In[ ]:


image1.shape


# In[9]:


print(high_frequencies-np.min(high_frequencies))


# In[ ]:




