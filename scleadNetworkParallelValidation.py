'''
Created on Oct 10, 2018

@author: deeplearning
'''

import tensorflow as tf
import constants as ct
from scleadNetworkParallelArchitecture import foward_propagation
from readImageFromTFRecord import readImageFromTFRecord
from writeAndReadFiles import readInfoFromFile
import time

def validate_network():
    dataSetSizeList = readInfoFromFile(ct.INFORMATION_PATH)
    validation_image_num = int(dataSetSizeList['validation'])
        
    curr_image_inputs=tf.placeholder(tf.float32, (1,ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL), 'curr_inputs')
    hist_image_inputs=tf.placeholder(tf.float32, (1,ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL), 'hist_inputs')
    label_inputs =tf.placeholder(tf.float32,(1,ct.CLASS_NUM), 'outputs')
    
    nn_output = foward_propagation(curr_image_inputs,hist_image_inputs,is_training=False)
    correct_prediction = tf.equal(tf.argmax(nn_output,1), tf.argmax(label_inputs,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    curr_img_tensor,hist_img_tensor,label_tensor= readImageFromTFRecord(ct.CATELOGS[2])
    curr_img_tensor=tf.reshape(curr_img_tensor,[1,ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL])
    hist_img_tensor = tf.reshape(hist_img_tensor,[1,ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL])
    label_tensor = tf.reshape(label_tensor,[1,ct.CLASS_NUM])
    saver = tf.train.Saver()
    with tf.Session() as sess :
         
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while(True):
            correct_prediction_list = []   
            ckpt = tf.train.get_checkpoint_state(ct.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                for _ in range(validation_image_num):
                    curr_img,hist_img,label = sess.run([curr_img_tensor,hist_img_tensor,label_tensor])
                    per_correct_prediction = sess.run(correct_prediction, 
                                                      feed_dict= {curr_image_inputs:curr_img,hist_image_inputs:hist_img,label_inputs:label})
                    correct_prediction_list.append(per_correct_prediction[0])
                correct_num = 0 
                for rst in correct_prediction_list:
                    correct_num+=rst
                accuracy_score = correct_num/len(correct_prediction_list)
                print('after %s iteration, the validation accuracy is %g'%(global_step,accuracy_score))
            else:
                print('no model')
#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#             print(sess.run(update_ops))
            print('running..........')
            time.sleep(200)
        coord.request_stop()
        coord.join(threads) 
                           
if __name__ == '__main__':
    with tf.device('/cpu:0'):            
        validate_network()      
                                                              
