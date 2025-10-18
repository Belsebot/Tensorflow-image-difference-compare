import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

image1_path = './kuvat/pic01.png'                                                  #first image
image2_path = './kuvat/pic02.png'                                                  #second image
image1 = tf.keras.utils.load_img(image1_path)                                      #keras image load
image2 = tf.keras.utils.load_img(image2_path)

image1_tensor = tf.convert_to_tensor(image1, dtype=tf.float32)                    #convert image information to tensors
image2_tensor = tf.convert_to_tensor(image2, dtype=tf.float32)

diff = tf.abs(image1_tensor - image2_tensor)                                      #get differences

diff_gray = tf.reduce_mean(diff,axis=-1)

threshold = 0.1
change_mask = tf.where(diff_gray > threshold, 1.0,0.0)

fig,axs = plt.subplots(1,3,figsize=(15,5))
axs[0].imshow(image1)
axs[0].set_title("Image1")
axs[0].axis('off')

axs[1].imshow(image2)
axs[1].set_title("Image2")
axs[1].axis('off')

axs[2].imshow(change_mask,cmap='hot')
axs[2].set_title("Changes")
axs[2].axis('off')

plt.tight_layout()
plt.savefig("Tensor_kuva.png")                                            #saves image where shows two orignal pictures and their differenceses
plt.show()
