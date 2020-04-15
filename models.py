import tensorflow as tf
import math
import numpy as np
import tensorflow.keras as K

from tensorflow.keras import layers
std_wd, std_ht, std_channel = 880, 680, 3
# taken from TF tutorial https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, inputs):
    q, k, v = inputs, inputs, inputs
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights

# taken from TF tutorial https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb
def scaled_dot_product_attention(q, k, v, mask=None):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name='graphConv_wt', trainable=True)

class graphConv(layers.Layer):
  def __init__(self, feat_, input_dim, act=tf.nn.relu, units=512 ): ## this will need to be updated based on total o/p shape
    super( graphConv , self).__init__()
    w_init = tf.random_normal_initializer()
    #w_init = tf.zeros_initializer()
    self.w = self.add_weight("kernel",
                                  shape=[ input_dim , units ])

    #self.w = tf.Variable( initial_value=w_init(shape=(input_dim, units), dtype='float32_ref' ), trainable=True ) #dtype='float32'),
    b_init = tf.zeros_initializer()
    self.b = self.add_weight("bias",
                                  shape=[ units, ])

    self.feat = feat_
    self.activation = act

  def call(self, inputs):
  #def call(self, inputs):
    neighbours = inputs
    print( self.feat )
    print( self.w )
    feat_w_dot = tf.matmul( self.feat, self.w ) # output = NUM nodes x units
    gcn_op = tf.matmul( neighbours, feat_w_dot ) # op = NUM nodes x units
    final_op = gcn_op + self.b # NUM nodes x units + units x 1
    return self.activation( final_op )

def modelForImageAndFeatManipulation(convInp, featInp, batch_size ):# inp_size is a tuple  w, h
    '''
    incomin - size of resized inp im
    outoin - downscaled version of image  ( down by 4 times ) tat will be sent to cropppin
    '''
    #cropInp = K.backend.placeholder(shape=( None, 4))

    ## tf.image.crop_and_resize takes in 2d tensor of feats but in our case it will be batch, num boxes and last Dim
    ## resape it
    cropInp = tf.split( featInp, [-1, 4], -1 )[-1] ## split featInp into ( batch, num boxes, 7 ) and ( batch, num boxes, 4 ) - last 4 are co -ords
    featInpForSplit = tf.reshape( cropInp, (-1, 4) ) ## reshape into ( batch x  boxes, 4 )

    numBoxes = tf.shape( cropInp )[1]
    tiled = tf.tile( tf.expand_dims( tf.range( batch_size ), -1 ), ( 1, numBoxes ) )
    cropIndices = tf.reshape( tiled, (-1,) )
    '''
    print( '----' )
    print( featInpForSplit )
    print( numBoxes )
    print( cropInp )
    print( cropIndices )
    '''

    #cropIndices = K.backend.placeholder( shape=(None,), dtype=tf.int32 )

    conv1 = K.layers.Conv2D( filters=64, kernel_size=(5, 5), padding='same', input_shape=( std_wd, std_ht, std_channel ), activation='relu' )( convInp )
    conv2 = K.layers.Conv2D( filters=32, kernel_size=(5, 5), padding='same', activation='relu' )( conv1 )
    maxpool = K.layers.MaxPooling2D(pool_size=4)( conv2 )

    ## this section is for redirection to crop
    #cropOP = tf.image.crop_and_resize( image=maxpool , boxes=cropInp , box_indices=cropIndices , crop_size=( 12 , 8 ) )
    cropOP = tf.image.crop_and_resize( image=maxpool , boxes=featInpForSplit , box_indices=cropIndices , crop_size=( 12 , 8 ) )
    cropTfOP = tf.split( cropOP, batch_size, axis=0 ) # tis sould be of sape batch , num boxes , crop sizes , channels
    finalOP = tf.stack( cropTfOP )
    #finalOP = tf.split( cropOP, batch_size , axis=0 ) # tis sould be of sape batch , num boxes , crop sizes , channels
    print( np.asarray( cropTfOP ) )
    print( finalOP )
    conv3Input = K.Input( tensor=finalOP )
    post_crop_conv3 = K.layers.Conv3D( filters=10, kernel_size=(1,5,5), activation='relu' )( conv3Input )
    print( '*********' )
    print( post_crop_conv3 )
    flatten = K.layers.Reshape( (numBoxes, 32*10 ), name='Reshape_1' )( post_crop_conv3 ) ## since conv1d takes 3-d input
    print( flatten )
    finalConv1D = K.layers.Conv1D( filters=100 , kernel_size=1, name='Conv1_1' )( flatten )
    print( '*********' )
    print( finalConv1D )
    finalOPForResize = K.layers.Reshape( (numBoxes, 100), name='Reshape_2' )( finalConv1D ) ## just so that it matchhes te paper :( 
    ## this section is for redirection to crop

    ## thhis section is for full conv of img
    maxpool_full_0 = K.layers.MaxPooling2D(pool_size=4)( conv1 )
    conv21 = K.layers.Conv2D( filters=32, kernel_size=(5, 5), padding='same', activation='relu' )( maxpool_full_0 )
    maxpool_full_1 = K.layers.MaxPooling2D(pool_size=4)( conv21 )
    conv22 = K.layers.Conv2D( filters=32, kernel_size=(5, 5), padding='same', activation='relu' )( maxpool_full_1 )
    maxpool_full_2 = K.layers.MaxPooling2D(pool_size=4)( conv22 )
    conv23 = K.layers.Conv2D( filters=32, kernel_size=(5, 5), padding='same', activation='relu' )( maxpool_full_2 )
    maxpool_full_3 = K.layers.MaxPooling2D(pool_size=4)( conv23 )
    flatten_main = K.layers.Flatten()( maxpool_full_3 )
    print( flatten_main )
    img_conv_op = K.layers.Dense( 32 )( flatten_main )
    im_repeat = K.backend.repeat( img_conv_op, tf.shape( cropInp  )[1] )
    ## thhis section is for full conv of img
    return finalOPForResize, im_repeat, featInp

def modelForOhe( oheInp ):
    conv1 = K.layers.Conv2D( filters=50 , kernel_size=(1, 3), padding='same', input_shape=( None, 30, 78 ), activation='relu' )( oheInp )
    conv2 = K.layers.Conv2D( filters=10 , kernel_size=(1, 3), padding='same', activation='relu' )( conv1 )
    #### paper indicates max pool BUT it only ives a 2D tensor as output 
    ### we lose the 'number of nodes' dimension ..so flatten and reshape
    #maxpool_full_ = K.layers.GlobalMaxPooling2D()( conv2 )

    maxpool_full_ = K.layers.MaxPooling2D(pool_size=4)( conv2 ) ## thhhhis is bull sit 
    ## if u max pool it will reduce W and H dimension whhhich in this case will be num boxes and max length
    ## so IF V wanna return batch, num boxes as it is, the last dimensoin can never be fixed !!!
    ## so eiter we will need to modify gcn input to take variable lenthh feat input or avoid maxpool
    print( conv2 )
    print( maxpool_full_ )
    returnOhe = K.layers.Reshape( ( tf.shape( oheInp )[1], 30*10 ), name='Reshape_3' )( conv2 )  ## thhis is done so tat final op is batch, num nodes, whatever is left
    #returnOhe = K.layers.Reshape( ( tf.shape( oheInp )[0],  tf.shape( oheInp )[1], 7*10 ), name='Reshape_3' )( maxpool_full_ )  ## thhis is done so tat final op is batch, num nodes, whatever is left
    print( returnOhe )
    return returnOhe

def finalLayer( mhaOP , totLabels ):
    print('FINAL LAYER --------', mhaOP )
    final_layer_conv1_0 = K.layers.Conv1D( filters=64 , padding='same' , kernel_size=1, activation='relu', name='Conv1_2' )( mhaOP )
    dropout_0 = K.layers.Dropout( 0.15 )( final_layer_conv1_0 )
    print( final_layer_conv1_0 )
    final_layer_conv1_1 = K.layers.Conv1D( filters=64 , padding='same', kernel_size=1, activation='relu', name='Conv1_3' )( dropout_0 )
    print( final_layer_conv1_1 )
    final_layer_conv1_2 = K.layers.Conv1D( filters=totLabels , padding='same' , kernel_size=1, activation='sigmoid', name='Conv1_4' )( final_layer_conv1_1 )
    print( final_layer_conv1_2 )
    return final_layer_conv1_2

class PossEnc( K.layers.Layer ):
    def __init__( self, op_dim=28 ):
        super( PossEnc , self).__init__()
        self.op_dim = op_dim

    def call(self, inputs):    
        num_boxes = tf.range( tf.shape( inputs )[1] ) ## number of boxes = number of timesteps needed for PE calc
        xArg = tf.cast( num_boxes, tf.float32 )
        return tf.map_fn( self.mapper, xArg )

    def mapper(self, elems):
        retVal = list()
        dim = self.op_dim
        for yy in range( dim ):
            #print( elems )
            if yy%2 == 0:
                pp = ( 1/pow( 10000, ( (2*yy)/dim ) ) )
                res = tf.math.sin( tf.math.multiply( pp, tf.cast( elems, tf.float32 ) ) )
            elif yy%2 == 1:
                pp = ( 1/pow( 10000, ( (2*yy)/dim ) ) )
                res = tf.math.cos( tf.math.multiply( pp, tf.cast( elems, tf.float32 ) ) )
            #print( res )
            retVal.append( res )
        return tf.stack( retVal )    

def applyPosEncodin( cropInp_pos ):
    peClass = PossEnc()
    classOP = peClass( cropInp_pos )
    finalPE_op = K.backend.repeat( classOP , tf.shape( cropInp_pos )[0] ) ## thhis willl op = boxes, batces, o/p dim BUT we need to reshape
    ## to batches, boxes, o/p dim
    '''
    we are repeating batch dimension since this is a fixed / non learnin PE 
    it will not change batch to batch ..more importantly it only depends on number of boxes in a 
    batc and not te exact value of inputs ..only sequence matters ..so its ok to repeat the same PE
    across batces 
    '''
    finalPE_op = tf.transpose( finalPE_op , perm=[1, 0, 2] )
    concatInpWithPE = tf.concat( axis=2, values=[ finalPE_op , cropInp_pos ] ) ## output will be num boxes X o/p dimension BUT need to add batc dimension !!
    return  concatInpWithPE
#MP = modelForImageAndFeatManipulation( ( 680, 880, 1 ) )
#print( MP )
#ohe = modelForOhe( (None,40,53) )
#print( ohe )
#kal = tf.placeholder( shape=(None,4), dtype=tf.float32 )
#mm, xArg = ( applyPosEncodin( kal ) )


#sess = tf.Session()
#print( sess.run( [ mm , xArg ], feed_dict={ kal :[ [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3] ]  }) )
