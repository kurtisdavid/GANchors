import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('./tf-models/slim')
from nets import inception
from preprocessing import inception_preprocessing

def inceptionv3(scope_name='g2'):
    g2 = tf.Graph() 
    with g2.as_default() as g:
        with g.name_scope(scope_name) as scope:
            module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/classification/1")
            input_ = tf.placeholder(tf.float32, shape=(299,299,3))
#            processed_image = inception_preprocessing.preprocess_image(input_, 299, 299, is_training=False)
            processed_image = tf.expand_dims(input_,0)
#            input_resize = tf.image.resize_images(input_,(299,299))
#            input_new = tf.reshape(input_resize,(-1,299,299,3))
            logits = module(processed_image)
            prediction = tf.argmax(logits,axis=1)
            proba_ = tf.nn.softmax(logits) 
            tensor_dict = {
                'input_': input_,
                'prediction': prediction,
                'processed': processed_image,
                'proba_': proba_,
                'logits': logits
            }
    return g2, tensor_dict 
