import tensorflow as tf
import tensorflow_hub as hub
'''
    Given a GAN G, create tensorflow graph that will allow us to optimize for z.
'''
def load_biggan():
    return hub.Module('https://tfhub.dev/deepmind/biggan-256/2')

def create_graph_mnist(WGAN,scope_name='g1',dim=28,z_dim=100,n_channels=1):
    g1 = tf.Graph()
    with g1.as_default() as g:
        with g.name_scope(scope_name) as scope:
             # image target + matrix A
            target = tf.placeholder(tf.float32, shape=(dim,dim,n_channels))
            num_pixels = tf.placeholder(tf.float32)
            filter = tf.placeholder(tf.float32, shape=(dim,dim,n_channels))

            z = tf.Variable(tf.random.uniform((1,z_dim),minval=-1,maxval=1))

def create_graph(truncation=0.5, scope_name='g1', G = None, dim = 256, z_dim=140, n_channels = 3):
    g1 = tf.Graph() 
    with g1.as_default() as g:
        with g.name_scope(scope_name) as scope:
            # image target + matrix A
            target = tf.placeholder(tf.float32, shape=(dim,dim,n_channels))
            num_pixels = tf.placeholder(tf.float32)
            filter = tf.placeholder(tf.float32, shape=(dim,dim,n_channels))
             
            # have learned z be an input and trainable
            init_z = tf.placeholder(tf.float32, shape=(1,z_dim))
            z = tf.Variable(truncation*tf.random.truncated_normal((1,z_dim)))
            assign_op = tf.assign(z,init_z)
            diff_z = z - init_z 
            # class conditional...(?)
            class_ = tf.placeholder(tf.int32)
            y = tf.one_hot(class_, 1000)
        #    y_index = tf.random.uniform([1], maxval=1000, dtype=tf.int32)
        #    y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label
            if G is None:
                G = load_biggan()
                sample = (tf.squeeze(G(dict(y=y, z=z, truncation=truncation))) + 1)/2 
            else:
                sample = tf.reshape(tf.squeeze(G(z)), (28,28))
            anchored = tf.multiply(filter,sample) 

            # compute loss 
            loss = tf.reduce_sum(tf.square(anchored-target)) / num_pixels
            opt = tf.train.MomentumOptimizer(0.1,0.9)
            min_ = opt.minimize(loss, var_list = [z])
           
            # obtain dictionary of tensors to extract results outside 
            tensor_dict = {
                'assign_op': assign_op,
                'target': target,
                'filter': filter,
                'class_': class_,
                'init_z': init_z,
                'diff_z': diff_z,
                'z': z,
                'anchored': anchored,
                'loss': loss,
                'opt': opt,
                'min_': min_,
                'sample': sample,
                'num_pixels': num_pixels
            }
    return g1, tensor_dict
