from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import glob


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_list = glob.glob(args.uhdb31_dir + '/*.png')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Read the file containing the pairs used for testing
            # pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            # paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
            print('Done')
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on UHDB31 images')

            for img_path in image_list:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(args.output_dir, base_name + '.txt')
                images = facenet.load_data([img_path], False, False, image_size)
                ft = sess.run(embeddings, feed_dict={images_placeholder: images, phase_train_placeholder: False})
                np.savetxt(save_path, ft)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('uhdb31_dir', type=str,
                        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('--output_dir', default='./uhdb31', help='output directory')
    parser.add_argument('model_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))