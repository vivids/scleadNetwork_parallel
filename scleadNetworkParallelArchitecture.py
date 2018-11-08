'''
Created on Oct 8, 2018

@author: deeplearning
'''
import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim
import constants as ct


class Block(collections.namedtuple('block', ['scope','unit_fn','args'])):
     'A named tuple describing a ResNet block'    
     
def subsample(inputs,factor,scope=None):
    if factor==1:
        return inputs
    else:
        return slim.max_pool2d(inputs,[1,1],stride = factor, scope = scope)
#         return slim.max_pool2d(inputs,[factor,factor],stride = factor, scope = scope)
    
def conv2d_same(inputs,num_outputs,kernel_size, stride,scope=None):
#     if stride == 1:
#         return slim.conv2d(inputs,num_outputs,kernel_size,stride=stride,padding='SAME',scope=scope)
#     else:
#         pad_total=kernel_size-1
#         pad_beg=pad_total//2
#         pad_end=pad_total-pad_beg
#         inputs=tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
    return slim.conv2d(inputs,num_outputs,kernel_size,stride = stride,padding='SAME', scope=scope)
  
@slim.add_arg_scope    
def stack_blocks_dense(net,blocks,outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope,'block',[net]) as sc:
            for i , unit in enumerate(block.args):
                with tf.variable_scope('unit%d'%(i+1), values=[net]):
                    unit_depth,unit_depth_bottleneck, unit_stride = unit
                    net =block.unit_fn(net,depth=unit_depth,depth_bottleneck=unit_depth_bottleneck,stride=unit_stride)
            net=slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net

def resnet_arg_scope(is_training=True, weight_decay=0.0001,batch_norm_decay=0.997,batch_norm_epsilon = 1e-5,batch_norm_scale=True):
    batch_norm_params= {
        'is_training':is_training,
        'decay':batch_norm_decay,
        'epsilon':batch_norm_epsilon,
        'scale':batch_norm_scale,
        'updates_collections':tf.GraphKeys.UPDATE_OPS}      
    
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],padding='SAME') as arg_sc:
                return arg_sc
            
@slim.add_arg_scope                     
def bottleneck(inputs,depth,depth_bottleneck,stride,outputs_collections=None,scope=None):
    with tf.variable_scope(scope,'bottleneck_v2',[inputs]) as sc:
        depth_in=slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact= slim.batch_norm(inputs,activation_fn=tf.nn.relu,scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
#             shortcut=slim.conv2d(shortcut,depth,[1,1],stride=stride,normalizer_fn=None, activation_fn=None,scope='shortcut')
        else:
            shortcut=slim.conv2d(inputs,depth,[1,1],stride=stride,normalizer_fn=None, activation_fn=None,scope='shortcut')
#             shortcut=slim.conv2d(inputs,depth,[3,3],stride=stride,normalizer_fn=None, activation_fn=None,scope='shortcut')
        
        residual = slim.conv2d(preact,depth_bottleneck,[1,1],stride = 1, scope = 'conv1')
        residual = conv2d_same(residual,depth_bottleneck,3,stride,scope='conv2')
        residual = slim.conv2d(residual,depth,[1,1],stride = 1,normalizer_fn=None, activation_fn=None, scope = 'conv3')
        output= shortcut+residual
         
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)
    
def resnet_v2(curr_image_inputs,hist_image_inputs,blocks,num_classes=None,global_pool=True,include_root_block=True,reuse=None,scope=None):
    with tf.variable_scope(scope+'_curr','resnet_v2_curr',[curr_image_inputs], reuse=reuse) as sc:
        end_points_collection_curr=sc.original_name_scope+'_end_points'
        with slim.arg_scope([slim.conv2d,bottleneck,stack_blocks_dense], outputs_collections= end_points_collection_curr):
            net_curr=curr_image_inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
#                     net=conv2d_same(net,128,7,stride=2,scope='conv1')
                        net_curr=conv2d_same(net_curr,128,7,stride=2,scope='conv1')
                net_curr=slim.max_pool2d(net_curr, [3,3], stride=2, scope='pool1')
            net_curr=stack_blocks_dense(net_curr,blocks)
            net_curr=slim.batch_norm(net_curr,activation_fn=tf.nn.relu,scope='postnorm')
            if global_pool:
                net_curr=tf.reduce_mean(net_curr, [1,2], name='pool5',keepdims=True)
             
#             end_points_curr = slim.utils.convert_collection_to_dict(end_points_collection_curr)
#             end_points_curr['prediction']=slim.softmax(net_curr,scope='predictions')       
             
    with tf.variable_scope(scope+'_hist','resnet_v2_hist',[hist_image_inputs], reuse=reuse) as sc:
        end_points_collection_hist=sc.original_name_scope+'_end_points'
        with slim.arg_scope([slim.conv2d,bottleneck,stack_blocks_dense], outputs_collections= end_points_collection_hist):
            net_hist=hist_image_inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
#                     net=conv2d_same(net,128,7,stride=2,scope='conv1')
                        net_hist=conv2d_same(net_hist,128,7,stride=2,scope='conv1')
                net_hist=slim.max_pool2d(net_hist, [3,3], stride=2, scope='pool1')
            net_hist=stack_blocks_dense(net_hist,blocks)
            net_hist=slim.batch_norm(net_hist,activation_fn=tf.nn.relu,scope='postnorm')
            if global_pool:
                net_hist=tf.reduce_mean(net_hist, [1,2], name='pool5',keepdims=True)
                        
#             end_points_hist = slim.utils.convert_collection_to_dict(end_points_collection_hist)
#             end_points_hist['prediction']=slim.softmax(net_hist,scope='predictions')      
    net =  tf.concat([net_curr,net_hist],3)
    net_shape=net.get_shape().as_list()
    net=tf.reshape(net, [net_shape[0],net_shape[3]]) 
    
    if num_classes is not None:
        net = slim.fully_connected(net, 512, scope='fc1')
        net = slim.fully_connected(net, num_classes, scope='fc2')
#                 net=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope='logits')
#                 end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#                 end_points['prediction']=slim.softmax(net,scope='predictions')

    return net
                
def resnet_v2_50(curr_image_inputs,hist_image_inputs, num_classes=None,global_pool=True, reuse=None,scope='resnet_v2_50'):
    blocks=[
        Block('block1',bottleneck,ct.BLOCK1),
        Block('block2',bottleneck,ct.BLOCK2),
        Block('block3',bottleneck,ct.BLOCK3),
        Block('block4',bottleneck,ct.BLOCK4)
        ]
    return resnet_v2(curr_image_inputs,hist_image_inputs,blocks,num_classes,global_pool,include_root_block=True, reuse=reuse,scope=scope)

def foward_propagation(curr_image_inputs,hist_image_inputs,is_training=True):
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net=resnet_v2_50(curr_image_inputs,hist_image_inputs,ct.CLASS_NUM)
        return net
