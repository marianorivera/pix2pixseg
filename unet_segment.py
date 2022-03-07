# unet_segment.py

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D,  SeparableConv2D, LeakyReLU
from tensorflow.keras.layers import MaxPooling2D, Dropout, UpSampling2D, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential


def unet_segment(filters_per_block,  num_classes, img_size, droprate=0.25):
    '''
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = np.array([num_channels, 256, 128, 64, 32])
                            where num_channels is the number of channels in the input image.
    num_classes         :   int, number of channels in the output
    img_size            :   (rows, cols, channels) ints,  input dimension
    droprate            :   float, dropout rate
    
    return
    A unet model for segmenting images into num_classes models
    
    --- Example ---          
    img_size = (160,160,3)
    filters_per_block = np.array([img_size[2], 256, 128, 64, 32])
    model_unet = Unet_segment(filters_per_block = filters_per_block,  
                              num_classes       = 3,
                              img_size          = img_size,
                              droprate          = 0.25,
                              )  
    MJJRM may 2021
    '''
    
    num_blocks  = len(filters_per_block)   
    drop        = droprate*tf.ones(num_blocks, tf.float32)
    num_blocks
    kernel_size = (3,3)
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    nm= 'encoder'
    Xdicc={}

    Xin  = tf.keras.layers.Input(shape=img_size, name="x_true")
    Xdicc[0] = Xin
    print(0, Xin.shape)
    
    # head
    X = Conv2D(10, kernel_size=kernel_size, padding='same', activation='relu', 
               name='encoder-conv0_0')(Xin) 
    # print(0, X.shape)
        
    numFilters=filters_per_block[0]
    for i in range(1,num_blocks):
        numFilters=filters_per_block[i]
        X = Conv2D(numFilters, kernel_size=kernel_size, padding='same', activation='relu', 
                   name='encoder-conv1_{}'.format(str(i)))(X) 
        X = Conv2D(numFilters, kernel_size=kernel_size, padding='same', activation='relu', 
                   name='encoder-conv2_{}'.format(str(i)))(X)
        X = Dropout(rate=drop[i], name='encoder-drop_{}'.format(str(i)))(X)
        X = MaxPooling2D(pool_size=(2,2), padding='valid', 
                         name='encoder-maxpool_{}'.format(str(i)))(X)
        Xdicc[i] = X
        # print(i, numFilters, Xdicc[i].shape) 

    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    Y=X
    for i in range(num_blocks-1,0,-1):
        if i>1:
            numFilters = filters_per_block[i] 
        else:
            numFilters = 128
        #print(i, numFilters, Y.shape, Xdicc[i-1].shape)
        Y = UpSampling2D(size=2, name='decoder-up_{}'.format(str(i)))(Y)  
        # print(i, numFilters, Y.shape, Xdicc[i].shape)
        Y = Concatenate(name='decoder-concat_{}'.format(str(i)))([Y, Xdicc[i-1]])
        Y = Conv2D(numFilters, kernel_size=(3,3), padding='same', activation='relu', 
                   name='decoder-conv2_{}'.format(str(i)))(Y)
        Y = Conv2D(numFilters, kernel_size=(3,3), padding='same', activation='relu', 
                   name='decoder-conv3_{}'.format(str(i)))(Y)
        Y = Dropout(rate=drop[i], name='decoder-drop_{}'.format(str(i)))(Y)

    # Tail 
    Y = Conv2D(32, kernel_size=(3,3), 
               padding='same', 
               activation=None,
               name='tail-2xch')(Y)

    Yout = Conv2D(num_classes, kernel_size=(1,1), 
               padding='same', 
               activation='softmax', 
               name='tail-last')(Y)

    # construye el modelo
    model = Model(inputs = Xin,  
                  outputs= [Yout], 
                  name   = 'Unet_segment')
    return model


def resunet_down_segment(filters_per_block,  num_classes, img_size, droprate=0.25):
    '''
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = np.array([num_channels, 256, 128, 64, 32])
                            where num_channels is the number of channels in the input image.
                            
    num_classes         :   int, number of channels in the output
    
    img_size            :   (rows, cols, channels) ints,  input dimension
    droprate            :   float, dropout rate
    
    return
    A unet model for segmenting images into num_classes models
    
    
    --- Example ---          
    
    img_size = (160,160,3)
    filters_per_block = np.array([img_size[2], 256, 128, 64, 32])
    
    model_resunet_down = resunet_down_segment(filters_per_block = filters_per_block,  
                              num_classes       = 3,
                              img_size          = img_size,
                              droprate          = 0.25,
                              )  
    
    MJJRM may 2021
    '''
    
    num_blocks  = len(filters_per_block)   
    drop        = droprate*tf.ones(num_blocks, 'float32')
    kernel_size = (3,3)
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    nm= 'encoder'
    Xdicc={}

    Xin  = Input(shape=img_size, name="x_true")
    Xdicc[0] = Xin
    #print(0, Xin.shape)
    
    # head
    X_skip = Conv2D(10, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv0_0')(Xin) 
    #print(0, X_skip.shape)
        
    numFilters=filters_per_block[0]
    for i in range(1,num_blocks):
        
        
        Xdicc[i] = X_skip
        numFilters=filters_per_block[i]
        
        X = SeparableConv2D(numFilters, kernel_size=kernel_size, padding='same', name='encoder-conv1_{}'.format(str(i)))(X_skip) 
        X = BatchNormalization()(X)
        X = Activation("relu")(X) 
        
        X = SeparableConv2D(numFilters, kernel_size=kernel_size, padding='same', name='encoder-conv2_{}'.format(str(i)))(X) 
        X = BatchNormalization()(X)
        X = Activation("relu")(X)          
        
        X = Dropout(rate=drop[i], name='encoder-drop_{}'.format(str(i)))(X)
        
        # residual aggregation 
        X = Concatenate(name='encoder-concat-residual-{}'.format(str(i)))([X,X_skip])
        X = Conv2D(numFilters, kernel_size=(1,1), padding='same', name='encoder-conv3_{}'.format(str(i)))(X) 
        X = BatchNormalization()(X)
        X = Activation("relu")(X)
        
        X_skip = MaxPooling2D(pool_size=(2,2), padding='valid', name='encoder-maxpool_{}'.format(str(i)))(X)
        
        #print(i, numFilters, Xdicc[i].shape) 

    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    Y=X_skip
    for i in range(num_blocks-1,0,-1):
        
        if i>1:
            numFilters = filters_per_block[i-1] 
        else:
            numFilters = 64
            
        Y = UpSampling2D(size=2, name='decoder-up_{}'.format(str(i)))(Y)  
        val = Y.shape  
        Y = Concatenate(name='decoder-concat_{}'.format(str(i)))([Y, Xdicc[i]])
        #print(i, numFilters, val, Xdicc[i].shape, '->', Y.shape, )
        
        Y = SeparableConv2D(numFilters, kernel_size=(3,3), padding='same', name='decoder-conv2_{}'.format(str(i)))(Y)
        Y = BatchNormalization()(Y)
        Y = Activation("relu")(Y) 
    
        Y = SeparableConv2D(numFilters, kernel_size=(3,3), padding='same', name='decoder-conv3_{}'.format(str(i)))(Y)
        Y = BatchNormalization()(Y)
        Y = Activation("relu")(Y) 
    
        Y = Dropout(rate=drop[i], name='decoder-drop_{}'.format(str(i)))(Y)

    # Tail 
    #print(Y.shape)
    Y = SeparableConv2D(32, kernel_size=(3,3), 
               padding='same', 
               activation=None,
               name='tail-2xch')(Y)

    #print(Y.shape)
    Yout = SeparableConv2D(num_classes, kernel_size=(1,1), 
               padding='same', 
               activation='softmax', 
               name='tail-last')(Y)
    #print(Yout.shape)
    
    # construye el modelo
    model = Model(inputs =Xin,  
                    outputs=[Yout], 
                    name   ='Unet_segment')
    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         

def BlockConv(filters, kernel_size, drop, lastname, padding='same', dilation=(1,1)):
    '''

    '''
    blockdown = Sequential(name='conv_block_'+lastname)

    blockdown.add(SeparableConv2D(filters    =filters, 
                                  kernel_size=kernel_size, 
                                  padding    =padding,
                                  dilation_rate=dilation[0]))
    blockdown.add(BatchNormalization())
    blockdown.add(LeakyReLU())

    blockdown.add(SeparableConv2D(filters    =filters, 
                                  kernel_size=kernel_size, 
                                  padding    =padding,
                                  dilation_rate=dilation[1]))
    blockdown.add(BatchNormalization())
    #blockdown.add(Activation("relu"))      
    return blockdown

def BlockBlendDown(filters, kernel_size, drop, lastname):
    '''
    
    '''
    block_blenddown = Sequential(name='blenddown_'+lastname)
    block_blenddown.add(SeparableConv2D(filters, 
                                    kernel_size = (3,3), 
                                    strides     = 2,         
                                    padding     = 'same'))    #downsample
    block_blenddown.add(BatchNormalization())
    block_blenddown.add(LeakyReLU())
    block_blenddown.add(Dropout(drop))
    return block_blenddown
        
def BlockBlend(filters, kernel_size, drop, lastname):
    '''
    
    '''
    block_blend = Sequential(name='blend_'+lastname)
    block_blend.add(SeparableConv2D(filters, 
                                    kernel_size = (3,3), 
                                    padding     = 'same'))
    block_blend.add(BatchNormalization())
    block_blend.add(LeakyReLU())
    block_blend.add(Dropout(drop))
    return block_blend
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
def resunet_segment(filters_per_block, num_classes, img_size, droprate=0.25, logits=False):
    '''
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = np.array([num_channels, 256, 128, 64, 32])
                            where num_channels is the number of channels in the input image.
                            
    num_classes         :   int, number of channels in the output
    
    img_size            :   (rows, cols, channels) ints,  input dimension
    droprate            :   float, dropout rate
    
    return
    A unet model for segmenting images into num_classes models
    
    
    --- Example ---          
    
    img_size = (160,160,3)
    filters_per_block = np.array([img_size[2], 256, 128, 64, 32])
    
    model_resunet_down = resunet_segment(filters_per_block = filters_per_block,  
                              num_classes       = 3,
                              img_size          = img_size,
                              droprate          = 0.25,
                              )  
    
    MJJRM may 2021
    '''
    
    num_blocks  = len(filters_per_block)   
    drop        = droprate*np.ones(num_blocks) #tf.ones(num_blocks, 'float32')
    kernel_size = (3,3)
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    nm= 'encoder'
    Xdicc={}

    Xin  = Input(shape=img_size, name="x_true")
    
    # head
    Xdicc[0] = Xin
    X = Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv0_0')(Xin) 
    
    for i in range(1,num_blocks):
        
        Xdicc[i]=X
        filters=filters_per_block[i]
        
        FX = BlockConv      (filters=filters, kernel_size=kernel_size, drop=drop[i], lastname='enc'+str(i))(X)
        X  = Concatenate    (name='encoder_concat_residual_{}'.format(str(i)))([X, FX])
        X  = BlockBlendDown (filters=filters, kernel_size=kernel_size,drop=drop[i], lastname= 'enc'+str(i))(X)
    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    Y=X
    for i in range(num_blocks-1,0,-1):
        
        if i>1:
            filters = filters_per_block[i-1] 
        else:
            filters = 64
            
        Y  = UpSampling2D (size=2, name='decoder-up_{}'.format(str(i)))(Y)  
        Y  = Concatenate  (name='decoder-concat_{}'.format(str(i)))([Y, Xdicc[i]])
        FY = BlockConv    (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec'+str(i))(Y)
        Y  = Concatenate  (name='decoder-concat-residual-{}'.format(str(i)))([Y,FY])
        Y  = BlockBlend   (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec'+str(i))(Y)
        
    # Tail 
    Y = SeparableConv2D(32, kernel_size=(3,3), 
               padding   = 'same', 
               activation= None,
               name      = 'tail-2xch')(Y)
    Y = Dropout(drop[0])(Y)

    Yout = SeparableConv2D(num_classes, kernel_size=(1,1), 
               padding   = 'same', 
               activation= None, 
               name      = 'tail-last')(Y)
    
    if not logits:
        Yout = Activation("softmax")(Yout)
            
    # construye el modelo
    model = Model(inputs =Xin,  
                    outputs=[Yout], 
                    name   ='ResUnet_segment')
    return model


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         

def resunet(filters_per_block, output_channels, img_size, droprate=0.25):
    '''
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = np.array([num_channels, 256, 128, 64, 32])
                            where num_channels is the number of channels in the input image.
                            
    output_channels     :   int, number of channels in the output
    
    img_size            :   (rows, cols, channels) ints,  input dimension
    droprate            :   float, dropout rate
    
    return
    A unet model for segmenting images into num_classes models
    
    
    --- Example ---          
    
    img_size = (160,160,3)
    filters_per_block = np.array([img_size[2], 256, 128, 64, 32])
    
    resunet_down = resunet(filters_per_block = filters_per_block,  
                           num_classes       = 3,
                           img_size          = img_size,
                           droprate          = 0.25,
                           )  
    
    MJJRM marzo 2022  (modificado)
    '''
    
    num_blocks  = len(filters_per_block)   
    drop        = droprate*np.ones(num_blocks) #tf.ones(num_blocks, 'float32')
    kernel_size = (3,3)
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    nm= 'encoder'
    Xdicc={}

    Xin  = Input(shape=img_size, name="x_true")
    
    # head
    Xdicc[0] = Xin
    X = Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv0_0')(Xin) 
    
    for i in range(1,num_blocks):
        
        Xdicc[i]=X
        filters=filters_per_block[i]
        
        FX = BlockConv      (filters=filters, kernel_size=kernel_size, drop=drop[i], lastname='enc'+str(i))(X)
        X  = Concatenate    (name='encoder_concat_residual_{}'.format(str(i)))([X, FX])
        X  = BlockBlendDown (filters=filters, kernel_size=kernel_size,drop=drop[i], lastname= 'enc'+str(i))(X)
    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    Y=X
    for i in range(num_blocks-1,0,-1):
        
        if i>1:
            filters = filters_per_block[i-1] 
        else:
            filters = 64
            
        Y  = UpSampling2D (size=2, name='decoder-up_{}'.format(str(i)))(Y)  
        Y  = Concatenate  (name='decoder-concat_{}'.format(str(i)))([Y, Xdicc[i]])
        FY = BlockConv    (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec'+str(i))(Y)
        Y  = Concatenate  (name='decoder-concat-residual-{}'.format(str(i)))([Y,FY])
        Y  = BlockBlend   (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec'+str(i))(Y)
        
    # Tail 
    Y = SeparableConv2D(output_channels, 
                       kernel_size=(3,3), 
                       padding   = 'same', 
                       activation= None,
                       name      = 'tail-2xch')(Y)
    Yout = Dropout(drop[0])(Y)

            
    # construye el modelo
    model = Model(inputs =Xin,  
                  outputs=[Yout], 
                  name   ='ResUnet')
    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         

def incepresunet_segment(filters_per_block,  num_classes, img_size, droprate=0.25):
    '''
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = np.array([num_channels, 256, 128, 64, 32])
                            where num_channels is the number of channels in the input image.
                            
    num_classes         :   int, number of channels in the output
    
    img_size            :   (rows, cols, channels) ints,  input dimension
    droprate            :   float, dropout rate
    
    return
    A unet model for segmenting images into num_classes models
    
    
    --- Example ---          
    
    img_size = (160,160,3)
    filters_per_block = np.array([img_size[2], 256, 128, 64, 32])
    
    model_resunet_down = incepresunet_segment(filters_per_block = filters_per_block,  
                                              num_classes       = 3,
                                              img_size          = img_size,
                                              droprate          = 0.25,
                                              )  
    
    MJJRM jul 2021
    '''
    
    num_blocks  = len(filters_per_block)   
    drop        = droprate*np.ones(num_blocks) #tf.ones(num_blocks, 'float32')
    kernel_size = (3,3)
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    nm= 'encoder'
    Xdicc={}

    Xin  = Input(shape=img_size, name="x_true")
    
    # head
    Xdicc[0] = Xin
    X = Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv0_0')(Xin) 
    
    for i in range(1,num_blocks):
        
        Xdicc[i]=X
        filters=filters_per_block[i]
        
        FX1 = BlockConv     (filters=filters//2, kernel_size=kernel_size, drop=drop[i], lastname='enc1_'+str(i), dilation=(1,1))(X)
        FX2 = BlockConv     (filters=filters//2, kernel_size=kernel_size, drop=drop[i], lastname='enc2_'+str(i), dilation=(1,2))(X)
        X   = Concatenate    (name='encoder_concat_residual_{}'.format(str(i)))([X, FX1, FX2])
        X   = BlockBlendDown (filters=filters, kernel_size=kernel_size,drop=drop[i], lastname= 'enc_'+str(i))(X)
    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    Y=X
    for i in range(num_blocks-1,0,-1):
        
        if i>1:
            filters = filters_per_block[i-1] 
        else:
            filters = 64
            
        Y  = UpSampling2D (size=2, name='decoder-up_{}'.format(str(i)))(Y)  
        Y  = Concatenate  (name='decoder-concat_{}'.format(str(i)))([Y, Xdicc[i]])
        FY = BlockConv    (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec_'+str(i))(Y)
        Y  = Concatenate  (name='decoder-concat-residual-{}'.format(str(i)))([Y,FY])
        Y  = BlockBlend   (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec_'+str(i))(Y)
        
    # Tail 
    #print(Y.shape)
    Y = SeparableConv2D(32, kernel_size=(3,3), 
               padding   ='same', 
               activation=None,
               name      ='tail-2xch')(Y)
    Y = Dropout(drop[0])(Y)

    #print(Y.shape)
    Yout = SeparableConv2D(num_classes, kernel_size=(1,1), 
               padding   ='same', 
               activation='softmax', 
               name      ='tail-last')(Y)
    #print(Yout.shape)
    
    # construye el modelo
    model = Model(inputs =Xin,  
                    outputs=[Yout], 
                    name   ='IncepResUnet_segment')
    return model


