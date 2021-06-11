##########################################################################
#                             IMPORTO LIBRARIES                          #
##########################################################################
import numpy      as np
import pandas     as pd
import glob
import os
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
diretorio_imagens_treino = diretorio_raiz + '/train_images'
diretorio_imagens_test   = diretorio_raiz + '/test_images'
diretorio_modelo         = diretorio_raiz + '/4_modelo_classificacao'

##########################################################################
#                             INICIO INGESTÃO DOS ARQUIVOS CHAVES        #
##########################################################################
base_depara_id_treino = pd.DataFrame(pd.read_csv(diretorio_modelo + '/1_de_para_json_treino_originals.csv',
                                                 encoding = 'utf8',
                                                 sep = ';',
                                                 error_bad_lines = True))
base_depara_id_teste  = pd.DataFrame(pd.read_csv(diretorio_modelo + '/3_de_para_json_teste_originals.csv',
                                                 encoding = 'utf8',
                                                 sep = ';',
                                                 error_bad_lines = True))

##########################################################################
#                             ANALIZO MENOR TAMANHO                      #
##########################################################################
min_width_treino  = base_depara_id_treino['Width'].min() #60
min_height_treino = base_depara_id_treino['Height'].min()#35
min_width_teste   = base_depara_id_teste['Width'].min()  #60
min_height_teste  = base_depara_id_teste['Height'].min() #60

##########################################################################
#                             DEFINO AJUSTES NAS IMAGENS                 #
##########################################################################
'''ESCOLHO PARA O TAMANHO O MAIOR TAMANHO MULTIPLO DE 2, 
QUE SEJA ABAIXO DO MINIMO'''
transf_treino     = ImageDataGenerator(rescale          = 1./255.0,
                                       validation_split = 0.4)
transf_teste      = ImageDataGenerator(rescale          = 1./255.0)
batch_size_treino = 20 #50
batch_size_valid  = 20 #50
train_generator = transf_treino.flow_from_directory(diretorio_imagens_treino,
                                                    shuffle     = True,
                                                    target_size = (40, 40),
                                                    batch_size  = batch_size_treino,
                                                    class_mode  = 'binary',
                                                    subset      = 'training')
valid_generator = transf_treino.flow_from_directory(diretorio_imagens_treino,
                                                    shuffle     = False,
                                                    target_size = (40, 40),
                                                    batch_size  = batch_size_treino,
                                                    class_mode  = 'binary',
                                                    subset      = 'validation')
test_generator  = transf_teste.flow_from_directory( diretorio_imagens_test,
                                                    shuffle     = False,
                                                    target_size = (40, 40),
                                                    batch_size  = batch_size_valid,
                                                    class_mode  = 'binary')
##########################################################################
#                             DEFINO ARQUITETURA DA REDE                 #
##########################################################################
model = models.Sequential() 
model.add(layers.Conv2D(32, (3, 3),  strides = (1, 1), padding = 'same', activation = 'relu', 
                        input_shape = (40, 40, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
initializer = initializers.RandomNormal(mean=0.0, stddev=0.05)
#model.add(layers.Dropout(0.6))
model.add(layers.Dense(128, activation = 'relu', kernel_initializer = initializer, bias_initializer   = 'zeros'))
#model.add(layers.Dropout(0.6))
model.add(layers.Dense(64,activation = 'relu',kernel_initializer = initializer))
#model.add(layers.Dropout(0.6))
model.add(layers.Dense(32, activation = 'relu', kernel_initializer = initializer))
#model.add(layers.Dropout(0.6))
model.add(layers.Dense(1,   
                       activation = 'sigmoid',
                       kernel_initializer = initializer))
model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.Adam (lr = 0.001),
              metrics = ['acc'])

##########################################################################
#                             TREINO O MODELO                            #
##########################################################################
history = model.fit_generator(train_generator,
                              steps_per_epoch  = train_generator.samples // batch_size_treino,
                              epochs           = 15,
                              validation_data  = valid_generator,
                              validation_steps = valid_generator.samples // batch_size_valid)

##########################################################################
#                             AVALIO LOSS & ACCURACY                     #
##########################################################################
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training Acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation Acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

##########################################################################
#                             SALVO MODELO                               #
##########################################################################
model.save(diretorio_modelo + '/model_cat_dog_epoch40_dropout_0.6')

##########################################################################
#                             KS E ROC                                   #
##########################################################################
VALID_PREDICT            = pd.DataFrame(columns = {'Label', 'Modelo'})
TESTE_PREDICT            = pd.DataFrame(columns = {'Label', 'Modelo'})
VALID_PREDICT['Label']   = valid_generator.labels
VALID_PREDICT['Modelo']  = model.predict(valid_generator)
TESTE_PREDICT['Label']   = test_generator.labels
TESTE_PREDICT['Modelo']  = model.predict(test_generator)

title_font = {'fontname' : 'Arial',
              'size'     : '17',
              'weight'   : 'bold'}
axis_font  = {'fontname' : 'Arial',
              'size'     : '12'}
lr_fpr_valid,  lr_tpr_valid,  _ = roc_curve(np.asarray(VALID_PREDICT['Label']),  np.asarray(VALID_PREDICT['Modelo']))
lr_fpr_teste,  lr_tpr_teste,  _ = roc_curve(np.asarray(TESTE_PREDICT['Label']),  np.asarray(TESTE_PREDICT['Modelo']))
plt.plot([0.0, 1.0], [0.0, 1.0], 'r--', linewidth = 0.5, label = 'Coin', color = 'black')
plt.plot(lr_fpr_valid,  
         lr_tpr_valid,  
         linewidth = 0.5, 
         label = 'Valid  - ' + str(round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo']))*100, 2)) + ' %', 
         color = 'red')
plt.plot(lr_fpr_teste,  
         lr_tpr_teste,  
         linewidth = 0.5, 
         label = 'Teste - ' + str(round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo']))*100, 2)) + ' %', 
         color = 'green')
plt.title('Curva ROC', title_font)
plt.xlabel('False Positive Rate', axis_font)
plt.ylabel('True Positive Rate', axis_font)
plt.legend()
plt.show()



