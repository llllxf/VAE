# -*- coding: utf-8 -*-
import tensorflow as tf

#高斯多层感知器作为编码器
def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope("gaussian_MLP_encoder"):
        #初始化
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        #隐藏层,计算x*w+b
        w0 = tf.get_variable('w0',[x.get_shape()[1],n_hidden],initializer=w_init)
        b0 = tf.get_variable('b0',[n_hidden],initializer=b_init)
        h0 = tf.matmul(x,w0)+b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0,keep_prob)

        #隐藏层,计算h0*w+b
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        #输出层
        wo = tf.get_variable('wo',[h1.get_shape()[1],n_output*2],initializer=w_init)
        bo = tf.get_variable('bo',[n_output*2],initializer=b_init)
        gaussian_params = tf.matmul(h1,wo)+bo

        #softplus即log( exp( features ) + 1),由于std必须是正的，所以采用softplus函数且加上一个极小正值
        mean = gaussian_params[:,:n_output] #取前半部分为mean
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:,n_output:])

    return mean,stddev

#Bernoulli MLP 作为解码器
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):
    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        #隐藏层,计算z*w0+b0
        w0 = tf.get_variable('w0',[z.get_shape()[1],n_hidden],initializer=w_init)
        b0 = tf.get_variable('b0',[n_hidden],initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        #隐藏层,计算h0*w1+b1
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        #输出层,根据激活函数sigmoid和h1得到n_output维的0-1值
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y


def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):

    #编码，得到隐藏变量z的u和v
    u, v = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)

    #重采样得到，利用 u+exp(v/2)*epsil 得到z
    z = u+v*tf.random_normal(tf.shape(u),0,1,dtype=tf.float32)

    #解码
    y = bernoulli_MLP_decoder(z,n_hidden,dim_img,keep_prob)
    y = tf.clip_by_value(y,1e-8,1-1e-8)

    #loss
    likelihood = tf.reduce_sum(x*tf.log(y)+(1-x)*tf.log(1-y),1)
    KL = 0.5 * tf.reduce_sum(tf.square(u) + tf.square(v) - tf.log(1e-8 + tf.square(v)) - 1, 1)
    marginal_likelihood = tf.reduce_mean(likelihood)
    KL_divergence = tf.reduce_mean(KL)
    ELBO = marginal_likelihood - KL_divergence
    loss = -ELBO

    return y, z, loss, -marginal_likelihood, KL_divergence

def decoder(z, dim_img, n_hidden):

    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)

    return y








