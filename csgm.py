import tensorflow as tf
import tensorflow_hub as hub

'''
    Given a GAN G, create tensorflow graph that will allow us to optimize for z.
'''
def create_graph(G,trunctation=0.0):
    # image target + matrix A
    target = tf.placeholder(tf.float32, shape=(3,256,256))
    filter = tf.placeholder(tf.float32, shape=(3,256,256))
    
    # have learned z be an input and trainable
    init_z = tf.placeholder(tf.float32, shape=(1,128))
    z = tf.Variable(tf.random.normal(1,128))
    z.assign(init_z)
    
    # class conditional...(?)
    class_ = tf.placeholder(tf.int32)
    y = tf.one_hot(class_, 1000)
    
    sample = module(dict(y=y, z=z, truncation=truncation))
    anchored = tf.math.multiply(filter,sample) # rewrite this to apply pixel wise dot

    # compute loss 
    loss = tf.losses.mean_squared_error(anchored, target)
    opt = tf.train.MomentumOptimizer(0.1,0.9)
    min_ = optimizer.minimize(loss)
   
    # obtain dictionary of tensors to extract results outside 
    tensor_dict = {
        'target': target,
        'filter': filter,
        'class_': class_,
        'init_z': init_z,
        'z': z,
        'loss': loss,
        'opt': opt,
        'min_': min_
    }
    return tensor_dict 
