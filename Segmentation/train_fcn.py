import os
import scipy as scp
import scipy.misc
import import_training_data
import cv2
import utils

import numpy as np
import tensorflow as tf

import fcn8_vgg
import loss

RESTORE_FROM_CHECKPOINT = False

from tensorflow.python.framework import ops

if os.path.isfile('p_images.npy'):
    p_images = np.load('p_images.npy')
    p_masks = np.load('p_masks.npy')
else: 
    images, masks, class_names = import_training_data.load_data()
    p_images, p_masks = import_training_data.preprocess_data(images, masks)
    np.save('p_images.npy', p_images)
    np.save('p_masks.npy', p_masks)

sess = tf.Session()

images = tf.placeholder("float")
labels = tf.placeholder("float")
batch_images = tf.expand_dims(images, 0)
dropout_keep_prob = tf.placeholder("float")

vgg_fcn = fcn8_vgg.FCN8VGG()
with tf.name_scope("content_vgg"):
    vgg_fcn.build(batch_images, train=True, debug=True, num_classes=4, dropout_keep_prob=dropout_keep_prob)

batch_labels = tf.expand_dims(labels, 0)
loss = loss.loss(vgg_fcn.upscore32, batch_labels, num_classes=4)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

print('Finished building Network.')

if RESTORE_FROM_CHECKPOINT:
   saver = tf.train.Saver()
   saver.restore(sess, './save/model.ckpt')
   print('Finished initializing variables.')
else:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('Finished initializing variables.')
    print('Training the network.')
    for e in range(10): # epochs
        for i in range(len(training_images)): # iterations
            feed_dict = {images: training_images[i], labels: training_annotations[i], dropout_keep_prob: 0.5} 
            tensors = [loss, train_step]
            loss_val, _ = sess.run(tensors, feed_dict=feed_dict)
            if (i % 50) == 0:
                print('Epoch:', e , '| Train step', i, '| Loss:', loss_val)
        
        total_test_loss = 0
        for i in range(len(test_images)):
            feed_dict = {images: test_images[i], labels: test_annotations[i], dropout_keep_prob: 1.0} 
            tensors = [vgg_fcn.pred, vgg_fcn.pred_up, vgg_fcn.upscore32, loss]
            down, up, upscore, loss_val = sess.run(tensors, feed_dict=feed_dict)
            total_test_loss += loss_val
        print('Epoch:', e , '| Average Test Loss:', total_test_loss / len(test_images))
    # Save network
    saver = tf.train.Saver()
    save_path = saver.save(sess, './save/model.ckpt')

TEST_IMAGES_NAMES = []
for test_image_name in TEST_IMAGES_NAMES:
    image = cv2.imread('./Data/' + test_image_name)
    feed_dict = {images: image, labels: test_annotations[0], dropout_keep_prob: 1.0} 
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up, vgg_fcn.upscore32]
    down, up, upscore = sess.run(tensors, feed_dict=feed_dict)
    up_color = utils.color_image(up[0], 4)
    scp.misc.imsave('./Test/' + test_image_name, up_color)
