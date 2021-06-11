##########################################################################
#                             IMPORTO LIBRARIES                          #
##########################################################################
import numpy      as np
import pandas     as pd
import glob
import os
import pickle
import matplotlib.pyplot as plt
from random                    import randint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras                     import models
from keras                     import layers
from keras                     import initializers
from keras                     import optimizers
from scipy.stats               import ks_2samp
from sklearn.metrics           import auc, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score

##########################################################################
#                             DEFINO DIRETORIOS                          #
##########################################################################
diretorio_raiz           = 'C:/Users/gusta/Desktop/Deep Learning'
diretorio_pickles        = diretorio_raiz + '/4_modelo_classificacao'
diretorio_avaliacao      = diretorio_raiz + '/6_Teste'

##########################################################################
#                             DEFINO AJUSTES NAS IMAGENS                 #
##########################################################################
transf_teste   = ImageDataGenerator(rescale     = 1./255.0)
test_generator = transf_teste.flow_from_directory(diretorio_avaliacao,
                                                  shuffle     = False,
                                                  target_size = (40, 40),
                                                  #batch_size  = 50,
                                                  class_mode  = None)

##########################################################################
#                             PREDICT                                    #
##########################################################################
#no programa do modelo:
#0 é gato
#1 é cachorro
model_1 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch8_dropout_0.6')
model_2 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch15_1_dropout_0.6')
model_3 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch15_2_dropout_0.6')
model_4 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch16_dropout_0.6')
model_5 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch30_dropout_0.6')
model_6 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch40_dropout_0.6')

TESTE_PREDICT = pd.DataFrame(columns = {'Modelo 1','Modelo 2', 
                                        'Modelo 3','Modelo 4', 
                                        'Modelo 5','Modelo 6', 
                                        'Ensemble','Resultado'})
TESTE_PREDICT['Modelo 1']  = pd.DataFrame(model_1.predict(test_generator))[0]
TESTE_PREDICT['Modelo 2']  = pd.DataFrame(model_2.predict(test_generator))[0]
TESTE_PREDICT['Modelo 3']  = pd.DataFrame(model_3.predict(test_generator))[0]
TESTE_PREDICT['Modelo 4']  = pd.DataFrame(model_4.predict(test_generator))[0]
TESTE_PREDICT['Modelo 5']  = pd.DataFrame(model_5.predict(test_generator))[0]
TESTE_PREDICT['Modelo 6']  = pd.DataFrame(model_6.predict(test_generator))[0]
TESTE_PREDICT['Ensemble']  = TESTE_PREDICT[['Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4', 'Modelo 5', 'Modelo 6']].max(axis = 1)
TESTE_PREDICT['Resultado'] = np.where(TESTE_PREDICT['Ensemble'] >= 0.375,
                                      'Cachorro',
                                      'Gato')
