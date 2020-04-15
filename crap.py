import tensorflow as tf
import tensorflow.keras as K
BATCH_SIZE = 2
NUM_BOXES = 5
IMAGE_HEIGHT = 20
IMAGE_WIDTH = 20
CHANNELS = 3
CROP_SIZE = (24, 24)


    #cropInp = K.backend.placeholder(shape=( None, 4))

    ## tf.image.crop_and_resize takes in 2d tensor of feats but in our case it will be batch, num boxes and last Dim
    ## resape it
convInp = K.backend.placeholder( shape=(None, None, None, 3) )
featInp = K.backend.placeholder( shape=(None, None, 6))


cropInp = tf.split( featInp, [-1, 4], -1 )[-1] ## split featInp into ( batch, num boxes, 7 ) and ( batch, num boxes, 4 ) - last 4 are co -ords
featInpForSplit = tf.reshape( cropInp, (-1, 4) ) ## reshape into ( batch x  boxes, 4 )

numBoxes = tf.shape( cropInp )[1]
batches = tf.shape( cropInp )[0]
tiled = tf.tile( tf.expand_dims( tf.range( batches ), -1 ), ( 1, numBoxes ) )
cropIndices = tf.reshape( tiled, (-1,) )
    #cropIndices = K.backend.placeholder( shape=(None,), dtype=tf.int32 )

cropOP = tf.image.crop_and_resize( image=convInp , boxes=featInpForSplit , box_indices=cropIndices , crop_size=( 12 , 8 ) )

finalOP = tf.split( cropOP, BATCH_SIZE, axis=0 )

import numpy as np
image = np.zeros(shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH,
CHANNELS) )
print( image.shape )
boxes = [ [ [ 1,1, 0.23, 0.34, 0.45, 0.67 ],[ 1,1, 0.3, 0.34, 0.45, 0.67   ] ], [ [ 1,1, 0.45, 0.34, 0.54, 0.67  ] ,[ 1,1, 0.23, 0.34, 0.88, 0.68  ]  ] ]

with tf.Session() as sess:

    op_, indix, tile, final = sess.run( [ cropOP, cropIndices, tiled, finalOP ], feed_dict={ convInp: image, featInp: boxes } )
    print( op_.shape )
    print( np.asarray(final).shape )
    print( indix )
    print( tile )
