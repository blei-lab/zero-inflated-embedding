from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
import collections
from scipy import sparse
import sys
from graph_builder import GraphBuilder
import warnings

def separate_valid(reviews, frac):
    review_size = reviews['scores'].shape[0]
    vind = np.random.choice(review_size, int(frac * review_size), replace=False)
    tind = np.delete(np.arange(review_size), vind)

    trainset = dict(scores=reviews['scores'][tind, :], atts=reviews['atts'][tind])
    validset = dict(scores=reviews['scores'][vind, :], atts=reviews['atts'][vind])
    
    return trainset, validset


def validate(valid_reviews, session, inputs, outputs):
    valid_size = valid_reviews['scores'].shape[0]
    ins_llh = np.zeros(valid_size)
    for iv in xrange(valid_size): 
        atts, indices, labels = generate_batch(valid_reviews, iv)
        if indices.size <= 1:
            raise Exception('in validation set: row %d has only less than 2 non-zero entries' % iv)
        feed_dict = {inputs['input_att']: atts, inputs['input_ind']: indices, inputs['input_label']: labels}
        ins_llh[iv] = session.run((outputs['llh']), feed_dict=feed_dict)
    
    mv_llh = np.mean(ins_llh)
    return mv_llh


def fit_emb(reviews, config):
    np.random.seed(27)

    # this options is only related to training speed. 
    config.update(dict(sample_ratio=0.1))

    use_valid_set = True 
    if use_valid_set:
        reviews, valid_reviews = separate_valid(reviews, 0.1)

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        
        builder = GraphBuilder()
        inputs, outputs, model_param = builder.construct_model_graph(reviews, config, init_model=None, training=True)

        optimizer = tf.train.AdagradOptimizer(0.05).minimize(outputs['objective'])
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()

        nprint = 5000
        val_accum = np.array([0.0, 0.0])
        train_logg = np.zeros([int(config['max_iter'] / nprint) + 1, 3]) 

        review_size = reviews['scores'].shape[0]
        for step in xrange(1, config['max_iter'] + 1):

            rind = np.random.choice(review_size)
            atts, indices, labels = generate_batch(reviews, rind)
            if indices.size <= 1: # neglect views with only one entry
                raise Exception('Row %d of the data has only one non-zero entry.' % rind)
            feed_dict = {inputs['input_att']: atts, inputs['input_ind']: indices, inputs['input_label']: labels}

            _, llh_val, obj_val, debug_val = session.run((optimizer, outputs['llh'], outputs['objective'], outputs['debugv']), feed_dict=feed_dict)
            val_accum = val_accum + np.array([llh_val, obj_val])

            # print loss every nprint iterations
            if step % nprint == 0 or np.isnan(llh_val) or np.isinf(llh_val):
                
                valid_llh = 0.0
                break_flag = False
                if use_valid_set:
                    valid_llh = validate(valid_reviews, session, inputs, outputs)
                    #if ivalid > 0 and valid_llh[ivalid] < valid_llh[ivalid - 1]: # performance becomes worse
                    #    print('validation llh: ', valid_llh[ivalid - 1], ' vs ', valid_llh[ivalid])
                    #    break_flag = True
                
                # record the three values 
                ibatch = int(step / nprint)
                train_logg[ibatch, :] = np.append(val_accum / nprint, valid_llh)
                val_accum[:] = 0.0 # reset the accumulater
                print("iteration[", step, "]: average llh, obj, and valid_llh are ", train_logg[ibatch, :])
                
                if np.isnan(llh_val) or np.isinf(llh_val):
                    print('Loss value is ', llh_val, ', and the debug value is ', debug_val)
                    raise Exception('Bad values')
   
                if break_flag:
                    break

        # save model parameters to dict
        model = dict(alpha=model_param['alpha'].eval(), 
                       rho=model_param['rho'].eval(), 
                     invmu=model_param['invmu'].eval(), 
                    weight=model_param['weight'].eval(), 
                       nbr=model_param['nbr'].eval())

        return model, train_logg

def evaluate_emb(reviews, model, config):

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        # construct model graph
        print('Building graph...')
        builder = GraphBuilder()
        inputs, outputs, model_param = builder.construct_model_graph(reviews, config, model, training=False)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        print('Initializing...')
        init.run()

        llh_array = [] 
        pos_llh_array = [] 
        review_size = reviews['scores'].shape[0]
        print('Calculating llh of instances...')
        for step in xrange(review_size):
            att, index, label = generate_batch(reviews, step)
            if index.size <= 1: # neglect views with only one entry
                continue
            feed_dict = {inputs['input_att']: att, inputs['input_ind']: index, inputs['input_label']: label}
            ins_llh_val, pos_llh_val = session.run((outputs['ins_llh'], outputs['pos_llh']), feed_dict=feed_dict)

            #if step == 0:
            #    predicts = session.run(outputs['debugv'], feed_dict=feed_dict)
            #    print('%d movies ' % predicts.shape[0])
            #    print(predicts)

            llh_array.append(ins_llh_val)
            pos_llh_array.append(pos_llh_val)

        
        llh_array = np.concatenate(llh_array, axis=0)
        pos_llh_array = np.concatenate(pos_llh_array, axis=0)

        
        return llh_array, pos_llh_array

def generate_batch(reviews, rind):
    atts = reviews['atts'][rind, :]
    _, ind, rate = sparse.find(reviews['scores'][rind, :])
    return atts, ind, rate 

