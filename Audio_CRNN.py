from keras.models import Model
import tensorflow as tf
from keras.layers import *

__all__ = ['Audio_NN']


def Audio_CRNN(_num_mel_scale, _num_time_len, _num_channel, _CL_1_kernal, _CL_2_kernal, _CL_3_kernal, _CL_4_kernal, _RNN_1_kernal, _RNN_2_kernal):
    Spectrogram_Input = Input(name= 'AudioInput',shape=(_num_mel_scale, _num_time_len, _num_channel)) #(None(1), 96, 1292, 1)
    CL_1 = Conv2D(_CL_1_kernal, kernel_size=(3, 3),padding='same', name = "A_CL_1")(Spectrogram_Input) #(None, 96, 1292, 128) 
    BN_1 = BatchNormalization()(CL_1)
    AL_1 = ELU()(BN_1)
    MP_1 = MaxPool2D(pool_size= (2,2), padding='same',name = "A_MPL_1")(AL_1) #(None, 48, 323, 128)
    DP_1 = Dropout(0.1)(MP_1)

    CL_2 = Conv2D(_CL_2_kernal, kernel_size=(3, 3),padding='same', name = "A_CL_2")(DP_1) # (None, 48, 323, 384) 
    BN_2 = BatchNormalization()(CL_2)
    AL_2 = ELU()(BN_2)
    MP_2 = MaxPool2D(pool_size= (3,3), padding='same',name = "A_MPL_2")(AL_2) # (None, 12, 64, 384) 
    DP_2 = Dropout(0.1)(MP_2)

    CL_3 = Conv2D(_CL_3_kernal, kernel_size=(3, 3),padding='same', name = "A_CL_3")(DP_2) #(None, 12, 64, 768) 
    BN_3 = BatchNormalization()(CL_3)
    AL_3 = ELU()(BN_3)
    MP_3 = MaxPool2D(pool_size= (4,4), padding='same',name = "A_MPL_3")(AL_3) #(None, 4, 8, 768)
    DP_3 = Dropout(0.1)(MP_3)

    CL_4 = Conv2D(_CL_4_kernal, kernel_size=(3, 3),padding='same', name = "A_CL_4")(DP_3) #(None, 4, 8, 2048) 
    BN_4 = BatchNormalization()(CL_4)
    AL_4 = ELU()(BN_4)
    MP_4 = MaxPool2D(pool_size= (4,4), padding='same',name = "A_MPL_4")(AL_4)#(None, 1, 1, 2048) 
    DP_4 = Dropout(0.1)(MP_4)

    AR = Permute((3,2,1))(DP_4)
    RS = Reshape((15, _CL_2_kernal))(DP_4)

    GRU_1 = GRU(_RNN_1_kernal, return_sequences=True, name='A_GRU_1')(RS)
    GRU_2 = GRU(_RNN_2_kernal, return_sequences=False, name='A_GRU_2')(GRU_1)
    
    return Spectrogram_Input, GRU_2



Audio_NN = {
    "CRNN" : Audio_CRNN,
}