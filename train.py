import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import cv2 
import dataFunc as DF
import models as MOD

std_wd, std_ht, std_channel = 880, 680, 3


class DataGenerator(K.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataDict, batch_size=32 ):
        'Initialization'
        self.data_dict_ = dataDict
        self.raw_img_rescaled = list()
        self.feat_ = list()
        self.inpt_pos_  = list()
        self.ohe_inp_ = list()
        self.labels_ = list()
        self.neighbours_ = list()
        self.batch_size = batch_size

        for key, val in self.data_dict_.items():
            ## val ( orderedNodes, neighBours, labels, OHE , features )
          img = cv2.imread( 'IM/'+key )
          rsz_im = cv2.resize( img, ( std_ht, std_wd ) )
          order_list = list( val[0].keys() )
          order_list.sort()
          tempOrder = list()
          for idx in order_list:
              arr = val[0][ idx ]
              tempOrder.append( [ arr[1], arr[2], arr[3], arr[4] ] )

          self.raw_img_rescaled.append( rsz_im )
          self.feat_.append( val[-1] )
          self.ohe_inp_.append( val[-2] )
          self.neighbours_.append( val[1] )
          self.labels_.append( val[2] )
          self.inpt_pos_.append( tempOrder )

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_dict_)) / (self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        img = self.raw_img_rescaled[ index*self.batch_size : (index+1)*self.batch_size ]
        feat_ = self.feat_[ index*self.batch_size : (index+1)*self.batch_size ]
        inpt_pos_ = self.inpt_pos_[ index*self.batch_size : (index+1)*self.batch_size ]
        ohe_inp_ = self.ohe_inp_[ index*self.batch_size : (index+1)*self.batch_size ]
        neighbours_ = self.neighbours_[ index*self.batch_size : (index+1)*self.batch_size ]

        y = self.labels_[ index*self.batch_size : (index+1)*self.batch_size ]
        # Find list of IDs
        # Generate data
        print('INPUT size of OE -- ', np.asarray( ohe_inp_ ).shape )
        return [ np.asarray( img ), np.asarray( feat_ ), np.asarray( inpt_pos_ ), np.asarray( ohe_inp_ ), np.asarray( neighbours_ ) ], np.asarray( y )


masterBatch, totalClasses = DF.loadData()

##test
trgBatch = masterBatch
valBatch = masterBatch

trgSeq = DataGenerator( trgBatch, batch_size=1 )
valSeq = DataGenerator( valBatch, batch_size=1 )

convInp = K.Input( shape=( std_wd, std_ht, std_channel ) )
featInp = K.Input( shape=( None, 11))

oheInp = K.Input( shape=( None, 30, 78 ) )
cropInp_pos = K.Input( shape=( None, 4 ), name='kansas' )
neighbours_Input = K.layers.Input( shape=(None,11) )

## modellingg starts
## FEATURE LAYERS
finalOPForResizeLayer , im_conv_layer , featInpLayer = MOD.modelForImageAndFeatManipulation( convInp, featInp,  batch_size=1 )
oheLayer = MOD.modelForOhe( oheInp )
posEncLayer = MOD.applyPosEncodin( cropInp_pos )

finalFeatLayer = K.layers.concatenate( [ oheLayer, posEncLayer, featInpLayer, finalOPForResizeLayer, im_conv_layer ] ) ## last dimension size = 275 ( + 200 if we avoid maxpool at ohe layer)
print( finalFeatLayer )
## FEATURE LAYERS

## graphhhhhhh convvvvvv
graphConvClass = MOD.graphConv( finalFeatLayer, input_dim=275+200 )
graphConvLayer = graphConvClass( neighbours_Input )
print( graphConvLayer )
## graphhhhhhh convvvvvv

postgraphConv_drop = K.layers.Dropout( 0.15 )( graphConvLayer )
postgraphConv1D_5 = K.layers.Conv1D( filters=128 , kernel_size=5, activation='relu', padding='same' )( postgraphConv_drop )
postgraphConv1D_1 = K.layers.Conv1D( filters=64 , kernel_size=1, activation='relu' )( postgraphConv1D_5 )
print( postgraphConv1D_1 )
## OP DIMENSION - last - 64
## Mha - attention 
mhaClass = MOD.MultiHeadAttention( d_model=64 , num_heads=8 )
mhaLayer , mhaOP = mhaClass( inputs=postgraphConv1D_1 )## typically we pass the same tensor in Q, K and V matrices
print( mhaLayer )
## Mha - attention 

## final layer 
flayer = MOD.finalLayer( mhaLayer, totalClasses )
print( flayer )
## final layer 
#print( finalOPForResize, im_repeat, featInp, oheLayer )
adam = K.optimizers.Adam(lr=0.0001)
#sgd = K.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
## np.asarray( img ), np.asarray( feat_ ), np.asarray( inpt_pos_ ), np.asarray( ohe_inp_ ), np.asarray( neighbours_ )
model = K.models.Model( [ convInp, featInp, cropInp_pos, oheInp, neighbours_Input ] , flayer , name='masaladosa')
model.compile( optimizer=adam , loss='binary_crossentropy', metrics=['accuracy'] )

model.fit_generator( generator=trgSeq , validation_data=valSeq , epochs=1 , verbose=1 )
