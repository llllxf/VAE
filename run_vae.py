# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import vae_tensorflow
import glob
import argparse
import mnist_data
import plot_util

IMAGE_SIZE_MNIST = 28
def parse_args():
    desc = "Tensorflow implementation of 'Variational AutoEncoder (VAE)'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    parser.add_argument('--add_noise', type=bool, default=False, help='Boolean for adding salt & pepper noise to input image')

    parser.add_argument('--dim_z', type=int, default='20', help='Dimension of latent vector', required = True)

    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--PRR', type=bool, default=True,
                        help='Boolean for plot-reproduce-result')

    parser.add_argument('--PRR_n_img_x', type=int, default=10,
                        help='Number of images along x-axis')

    parser.add_argument('--PRR_n_img_y', type=int, default=10,
                        help='Number of images along y-axis')

    parser.add_argument('--PRR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR', type=bool, default=False,
                        help='Boolean for plot-manifold-learning-result')

    parser.add_argument('--PMLR_n_img_x', type=int, default=20,
                        help='Number of images along x-axis')

    parser.add_argument('--PMLR_n_img_y', type=int, default=20,
                        help='Number of images along y-axis')

    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR_z_range', type=float, default=2.0,
                        help='Range for unifomly distributed latent vector')

    parser.add_argument('--PMLR_n_samples', type=int, default=5000,
                        help='Number of samples in order to get distribution of labeled data')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

    # --results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)

    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except:
        print('PRR must be boolean type')
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')

        # --PRR_resize_factor
        try:
            assert args.PRR_resize_factor > 0
        except:
            print('PRR : resize factor for each displayed image must be positive')

    # --PMLR
    try:
        assert args.PMLR == True or args.PMLR == False
    except:
        print('PMLR must be boolean type')
        return None

    if args.PMLR == True:
        try:
            assert args.dim_z == 2
        except:
            print('PMLR : dim_z must be two')

        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')

        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')

        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')

        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

    return args


def main(args):

    """ parameters """
    RESULTS_DIR = args.results_path

    # network architecture
    ADD_NOISE = args.add_noise

    n_hidden = args.n_hidden
    dim_img = IMAGE_SIZE_MNIST**2  # number of pixels for a MNIST image
    dim_z = args.dim_z

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    # Plot
    PRR = args.PRR                              # Plot Reproduce Result
    PRR_n_img_x = args.PRR_n_img_x              # number of images along x-axis in a canvas
    PRR_n_img_y = args.PRR_n_img_y              # number of images along y-axis in a canvas
    PRR_resize_factor = args.PRR_resize_factor  # resize factor for each image in a canvas

    PMLR = args.PMLR                            # Plot Manifold Learning Result
    PMLR_n_img_x = args.PMLR_n_img_x            # number of images along x-axis in a canvas
    PMLR_n_img_y = args.PMLR_n_img_y            # number of images along y-axis in a canvas
    PMLR_resize_factor = args.PMLR_resize_factor# resize factor for each image in a canvas
    PMLR_z_range = args.PMLR_z_range            # range for random latent vector
    PMLR_n_samples = args.PMLR_n_samples        # number of labeled samples to plot a map from input data space to the latent space

    """ prepare MNIST data """

    train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
    n_samples = train_size

    x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for PMLR
    z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

    # network architecture
    y, z, loss, neg_marginal_likelihood, KL_divergence = vae_tensorflow.autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob)

    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    """ training """

    if PRR:
        PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_MNIST,
                                                    IMAGE_SIZE_MNIST, PRR_resize_factor)

        x_PRR = test_data[0:PRR.n_tot_imgs, :]

        x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
        PRR.save_images(x_PRR_img, name='input.jpg')

        if ADD_NOISE:
            x_PRR = x_PRR * np.random.randint(2, size=x_PRR.shape)
            x_PRR += np.random.randint(2, size=x_PRR.shape)

            x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
            PRR.save_images(x_PRR_img, name='input_noise.jpg')
        # Plot for manifold learning result
    if PMLR and dim_z == 2:

        PMLR = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, PMLR_n_img_x, PMLR_n_img_y, IMAGE_SIZE_MNIST,
                                                            IMAGE_SIZE_MNIST, PMLR_resize_factor, PMLR_z_range)

        x_PMLR = test_data[0:PMLR_n_samples, :]
        id_PMLR = test_labels[0:PMLR_n_samples, :]

        if ADD_NOISE:
            x_PMLR = x_PMLR * np.random.randint(2, size=x_PMLR.shape)
            x_PMLR += np.random.randint(2, size=x_PMLR.shape)

        decoded = vae_tensorflow.decoder(z_in, dim_img, n_hidden)

    total_batch = int(n_samples / batch_size)
    min_tot_loss = 1e99

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})

        for epoch in range(n_epochs):
            # Random shuffling
            np.random.shuffle(train_total_data)

            #割掉标签
            train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]

            for i in range(total_batch):
                offset = (i*batch_size)%(n_samples)
                batch_xs_input = train_data_[offset:(offset+batch_size),:]

                batch_xs_target = batch_xs_input

                if ADD_NOISE:
                    batch_xs_input = batch_xs_input*np.random.randint(2,size=batch_xs_input.shape)
                    batch_xs_input += np.random.randint(2,size=batch_xs_input.shape)

                _, tot_loss, loss_likelihood,loss_divergence = sess.run((train_op, loss, neg_marginal_likelihood, KL_divergence),feed_dict={x_hat:batch_xs_input, x:batch_xs_target, keep_prob:0.9})

                print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                epoch, tot_loss, loss_likelihood, loss_divergence))

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)



















