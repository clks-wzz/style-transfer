# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:27:34 2017

@author: Zezheng Wang 
"""
import tensorflow as tf
import scipy.io as sio 
import numpy as np
import os
from PIL import Image 
#import matplotlib.pyplot as plt

train_epoch=6666
learning_rate=1
alpha=10e-3
beta=1
style=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
content=['conv4_2'];
vgg_model_path='./model/imagenet-vgg-verydeep-16.mat';
VGG_MEAN = [103.939, 116.779, 123.68]
save_name='./res.jpg'
#deal image
image_size=[1,256,256,3]
image_path='./'
image_content=Image.open(os.path.join(image_path, 'content.jpg'))
res_size=image_content.size
image_content=image_content.resize([256,256])
image_content=np.float32(image_content)
image_content[0:][0:][0]-=VGG_MEAN[0]
image_content[0:][0:][1]-=VGG_MEAN[1]
image_content[0:][0:][2]-=VGG_MEAN[2]
image_content=image_content.reshape(image_size)

image_style=Image.open(os.path.join(image_path, 'style.jpg'))
image_style=image_style.resize([256, 256])
image_style=np.float32(image_style)
image_style[0:][0:][0]-=VGG_MEAN[0]
image_style[0:][0:][1]-=VGG_MEAN[1]
image_style[0:][0:][2]-=VGG_MEAN[2]
image_style=image_style.reshape(image_size)

x1=tf.constant(image_content) #content
x2=tf.constant(image_style) #style
x1=tf.reshape(x1, image_size)
x2=tf.reshape(x2, image_size)
goal=tf.Variable(image_content)

vgg=sio.loadmat(vgg_model_path) 
vgg=vgg['layers']
#vgg[0][i][0][0][2][0][0] the params of ith layer
#vgg[0][i][0][0][1]       the type of ith layer
#vgg[0][i][0][0][0]       the name of ith layer
weights={}
bias={}
for i in range(vgg.shape[1]):
    if(vgg[0][i][0][0][1][0]=='conv'):
        weights[vgg[0][i][0][0][0][0]]=vgg[0][i][0][0][2][0][0]
        if vgg[0][i][0][0][2][0].size==2:
            bias[vgg[0][i][0][0][0][0]]=np.reshape(vgg[0][i][0][0][2][0][1], vgg[0][i][0][0][2][0][1].size)

def conv2d(x, w, b, strides=1):
    y1=tf.nn.conv2d(x,w,strides=[1, strides, strides, 1], padding='SAME')
    y2=tf.nn.bias_add(y1,b)
    out=tf.nn.relu(y2)
    return out

def pooling2d(x, strides=2, method='MAX'):
    if method=='MAX':
        y1=tf.nn.max_pool(x, ksize=[1, strides, strides, 1], strides=[1, strides, strides, 1], padding='SAME')
    elif method=='AVG':
        y1=tf.nn.avg_pool(x, ksize=[1, strides, strides, 1], strides=[1, strides, strides, 1], padding='SAME')
    else:
        print('pooling method error!')
    return y1
    
def content_loss(layerx, layery):
    loss=0.5*tf.reduce_sum(tf.pow(layerx-layery,2))
    return loss

def style_loss(layerx, layery):
    h=layerx.get_shape().as_list()[3]
    w=layerx.get_shape().as_list()[1]*layerx.get_shape().as_list()[2]  
    x1=tf.reshape(layerx, [w, h])
    x2=tf.transpose(x1)
    x=tf.matmul(x2, x1)
    y1=tf.reshape(layery, [w, h])
    y2=tf.transpose(y1)
    y=tf.matmul(y2, y1)
    loss= tf.multiply(1./(4* h**2 * w**2),tf.reduce_sum(tf.pow(x-y, 2)))
    return loss
    
def conv_net_style(x, weights, bias):
    res={}
    net=conv2d(x, tf.constant(weights['conv1_1']), tf.constant(bias['conv1_1']))
    res['conv1_1']=net;
    net=conv2d(net, tf.constant(weights['conv1_2']), tf.constant(bias['conv1_2']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv2_1']), tf.constant(bias['conv2_1']))
    res['conv2_1']=net;
    net=conv2d(net, tf.constant(weights['conv2_2']), tf.constant(bias['conv2_2']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv3_1']), tf.constant(bias['conv3_1']))
    res['conv3_1']=net;
    net=conv2d(net, tf.constant(weights['conv3_2']), tf.constant(bias['conv3_2']))
    net=conv2d(net, tf.constant(weights['conv3_3']), tf.constant(bias['conv3_3']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv4_1']), tf.constant(bias['conv4_1']))
    res['conv4_1']=net;
    net=conv2d(net, tf.constant(weights['conv4_2']), tf.constant(bias['conv4_2']))
    net=conv2d(net, tf.constant(weights['conv4_3']), tf.constant(bias['conv4_3']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv5_1']), tf.constant(bias['conv5_1']))
    res['conv5_1']=net;
    net=conv2d(net, tf.constant(weights['conv5_2']), tf.constant(bias['conv5_2']))
    net=conv2d(net, tf.constant(weights['conv5_3']), tf.constant(bias['conv5_3']))
    return res

def conv_net_content(x, weights, bias):
    res={}
    net=conv2d(x, tf.constant(weights['conv1_1']), tf.constant(bias['conv1_1']))
    net=conv2d(net, tf.constant(weights['conv1_2']), tf.constant(bias['conv1_2']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv2_1']), tf.constant(bias['conv2_1']))
    net=conv2d(net, tf.constant(weights['conv2_2']), tf.constant(bias['conv2_2']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv3_1']), tf.constant(bias['conv3_1']))
    net=conv2d(net, tf.constant(weights['conv3_2']), tf.constant(bias['conv3_2']))
    net=conv2d(net, tf.constant(weights['conv3_3']), tf.constant(bias['conv3_3']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv4_1']), tf.constant(bias['conv4_1']))
    net=conv2d(net, tf.constant(weights['conv4_2']), tf.constant(bias['conv4_2']))
    res['conv4_2']=net
    net=conv2d(net, tf.constant(weights['conv4_3']), tf.constant(bias['conv4_3']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv5_1']), tf.constant(bias['conv5_1']))
    net=conv2d(net, tf.constant(weights['conv5_2']), tf.constant(bias['conv5_2']))
    net=conv2d(net, tf.constant(weights['conv5_3']), tf.constant(bias['conv5_3']))
    return res

def conv_net(x, weights, bias, content_net, style_net):
    cost=tf.Variable(tf.cast(0, tf.float32))
    net=conv2d(x, tf.constant(weights['conv1_1']), tf.constant(bias['conv1_1']))
    cost=tf.add(cost, beta*style_loss(net, style_net['conv1_1']))      #style
    net=conv2d(net, tf.constant(weights['conv1_2']), tf.constant(bias['conv1_2']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv2_1']), tf.constant(bias['conv2_1']))
    cost=tf.add(cost, beta*style_loss(net, style_net['conv2_1']))      #style
    net=conv2d(net, tf.constant(weights['conv2_2']), tf.constant(bias['conv2_2']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv3_1']), tf.constant(bias['conv3_1']))
    cost=tf.add(cost, beta*style_loss(net, style_net['conv3_1']))      #style
    net=conv2d(net, tf.constant(weights['conv3_2']), tf.constant(bias['conv3_2']))
    net=conv2d(net, tf.constant(weights['conv3_3']), tf.constant(bias['conv3_3']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv4_1']), tf.constant(bias['conv4_1']))
    cost=tf.add(cost, beta*style_loss(net, style_net['conv4_1']))      #style
    net=conv2d(net, tf.constant(weights['conv4_2']), tf.constant(bias['conv4_2']))
    cost=tf.add(cost, alpha*content_loss(net, content_net['conv4_2']))      #content
    net=conv2d(net, tf.constant(weights['conv4_3']), tf.constant(bias['conv4_3']))
    net=pooling2d(net,method='AVG')
    net=conv2d(net, tf.constant(weights['conv5_1']), tf.constant(bias['conv5_1']))
    cost=tf.add(cost, beta*style_loss(net, style_net['conv5_1']))      #style
    net=conv2d(net, tf.constant(weights['conv5_2']), tf.constant(bias['conv5_2']))
    net=conv2d(net, tf.constant(weights['conv5_3']), tf.constant(bias['conv5_3']))
    return cost

content_net=conv_net_content(x1, weights, bias)
style_net=conv_net_style(x2, weights, bias)

cost=conv_net(goal, weights, bias, content_net, style_net)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_epoch):
        _, c, res=sess.run([optimizer, cost, goal])
        print('epoch=', i+1, ':', 'loss=', c)
        res=res.reshape([256,256,3])
        res[0:][0:][0]+=VGG_MEAN[0]
        res[0:][0:][1]+=VGG_MEAN[1]
        res[0:][0:][2]+=VGG_MEAN[2]
        new_im = Image.fromarray(res.astype(np.uint8))  
        new_im.save(save_name)
        fp = open(save_name,'r')
        fp.close()
        

