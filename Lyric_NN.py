from keras.models import Model
import tensorflow as tf
from keras.layers import *

__all__ = ['Lyric_NN']


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim



def Lyric_CRARNN(_num_lines, _num_words, _num_WEdim, _num_LyricCNN_kernalSize, _num_LyricCNN_kernalnum, _num_WGRU, _num_LGRU):
    Lines_Input = Input(name= 'TextInput',shape=(_num_words, _num_WEdim,)) #(None,25,100,1)
    LineFeature_List = []
    for i in range(_num_LyricCNN_kernalSize):
        CL   =  Conv1D(filters=_num_LyricCNN_kernalnum, padding='same', kernel_size = i+3, activation='relu', name='W_CL_'+str(i+3))(Lines_Input) 
        LineFeature_List.append(CL)
    LineFeature = concatenate(LineFeature_List)
    W_RNN_F = GRU(_num_WGRU, return_sequences=True, name='W_GRU_F')(LineFeature)
    W_RNN_B = GRU(_num_WGRU, return_sequences=True, name='W_GRU_B', go_backwards=True)(LineFeature)
    RNN_Concat = concatenate([W_RNN_F,W_RNN_B])
    AT_L = Attention(_num_words)(RNN_Concat)
    TextCNNMode = Model(inputs = Lines_Input, outputs = AT_L)
    
    Text_Input = Input(name='LyricInput', shape=(_num_lines,_num_words, _num_WEdim)) #(batch_size, timestep, 25, 100 ,1)
    TD_CNN = TimeDistributed(TextCNNMode)(Text_Input)
    Mask_L = Masking(mask_value=0.)(TD_CNN)
    GRU_F = GRU(_num_LGRU, return_sequences=False, name='L_GRU_F')(Mask_L)
    GRU_B = GRU(_num_LGRU, return_sequences=False, name='L_GRU_B', go_backwards=True)(Mask_L)
    LF = concatenate([GRU_F, GRU_B])
    
    return Text_Input, LF

def Lyric_RNN(_num_lines, _num_words, _num_WEdim, _num_LyricCNN_kernalSize, _num_LyricCNN_kernalnum, _num_WGRU, _num_LGRU):
    Text_Input = Input(name='LyricInput', shape=(_num_lines, _num_words, _num_WEdim)) #(batch_size, timestep, 25, 100 ,1)
    RL = Reshape(target_shape=(_num_lines*_num_words, _num_WEdim))(Text_Input)
    Mask_L = Masking(mask_value=0.)(RL)
    GRU_F = GRU(_num_WEdim, return_sequences=False, name='GRU_forward')(Mask_L)
    GRU_B = GRU(_num_WEdim, return_sequences=False, name='GRU_backward', go_backwards=True)(Mask_L)
    LF = concatenate([GRU_F, GRU_B])
    
    return Text_Input, LF

def Lyric_HARNN(_num_lines, _num_words, _num_WEdim, _num_LyricCNN_kernalSize, _num_LyricCNN_kernalnum, _num_WGRU, _num_LGRU):
    Lines_Input = Input(name= 'TextInput',shape=(_num_words, _num_WEdim))
    W_Mask = Masking(mask_value=0.)(Lines_Input)
    W_GRU_F = GRU(_num_WGRU, return_sequences=True, name='W_GRU_F')(W_Mask)
    W_GRU_B = GRU(_num_WGRU, return_sequences=True, name='W_GRU_B', go_backwards=True)(W_Mask)
    W_LF = concatenate([W_GRU_F, W_GRU_B])
    W_AT = Attention(_num_words)(W_LF)
    TextWordARNN = Model(inputs = Lines_Input, outputs = W_AT)
    
    Text_Input = Input(name='LyricInput', shape=(_num_lines, _num_words, _num_WEdim)) #(batch_size, timestep, 25, 100 ,1)
    TD_ARNN = TimeDistributed(TextCNNMode)(Text_Input)
    L_Mask = Masking(mask_value=0.)(TD_ARNN)
    L_GRU_F = GRU(_num_LGRU, return_sequences=True, name='L_GRU_F')(L_Mask)
    L_GRU_B = GRU(_num_LGRU, return_sequences=True, name='L_GRU_B', go_backwards=True)(L_Mask)
    L_LF = concatenate([L_GRU_F, L_GRU_B])
    L_AT = Attention(_num_lines)(L_LF)

    return Text_Input, L_AT

def Lyric_CNN(_num_lines, _num_words, _num_WEdim, _num_LyricCNN_kernalSize, _num_LyricCNN_kernalnum, _num_WGRU, _num_LGRU):
    Text_Input = Input(name='LyricInput', shape=(_num_lines, _num_words, _num_WEdim))
    RL = Reshape(target_shape=(_num_lines*_num_words, _num_WEdim))(Text_Input)

    LineFeature_List = []
    for i in range(_num_LyricCNN_kernalSize):
        #result_shape = (timestep, sentencelen(num of CNN result for each kernal), WE_dim, num_kernal) = (None,25,100,32)
        CL   =  Conv1D(filters=_num_LyricCNN_kernalnum, padding='same', kernel_size = i+3, activation='relu', name='W_CL_'+str(i+3))(RL) 
        DP = Dropout(0.5)(CL)
        #max pooling for every kernal, result_shape = (None, 1, 1, 32)
        MPL = MaxPooling1D(pool_size=2, strides=None, padding='valid', name='MP'+str(i+3))(DP)
        #MP_shape =  tf.shape(MPL)
        #flatten for matching GRU input size
        FL  = Flatten()(MPL) #result = (timestep size , )
        LineFeature_List.append(FL)
    LineFeature = concatenate(LineFeature_List)
    
    return Text_Input, LineFeature

    
Lyric_NN = {
    "CRARNN" : Lyric_CRARNN,
    "RNN": Lyric_RNN,
    "HARNN": Lyric_HARNN,
    "CNN": Lyric_CNN
}