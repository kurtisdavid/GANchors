import tensorflow as tf
import tensorflow_hub as hub

'''
    Given a GAN G, create tensorflow graph that will allow us to optimize for z.
'''
def load_gan():
    return hub.Module('https://tfhub.dev/deepmind/biggan-256/2')

def create_graph(G,truncation=0.5):
    # image target + matrix A
    target = tf.placeholder(tf.float32, shape=(256,256,3))
    filter = tf.placeholder(tf.float32, shape=(256,256,3))
    
    # have learned z be an input and trainable
    init_z = tf.placeholder(tf.float32, shape=(1,140))
    z = tf.Variable(tf.random_normal((1,140)))
    z.assign(init_z)
    
    # class conditional...(?)
    class_ = tf.placeholder(tf.int32)
    y = tf.one_hot(class_, 1000)
#    y_index = tf.random.uniform([1], maxval=1000, dtype=tf.int32)
#    y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label
    
    sample = tf.squeeze(G(dict(y=y, z=z, truncation=truncation)))
    anchored = tf.multiply(filter,sample) # rewrite this to apply pixel wise dot

    # compute loss 
    loss = tf.losses.mean_squared_error(anchored, target)
    opt = tf.train.MomentumOptimizer(0.1,0.9)
    min_ = opt.minimize(loss, var_list = [z])
   
    # obtain dictionary of tensors to extract results outside 
    tensor_dict = {
        'target': target,
        'filter': filter,
        'class_': class_,
        'init_z': init_z,
        'z': z,
        'loss': loss,
        'opt': opt,
        'min_': min_,
        'sample': sample
    }
    return tensor_dict
