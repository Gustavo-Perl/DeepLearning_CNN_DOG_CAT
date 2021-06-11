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
diretorio_imagens_treino = diretorio_raiz + '/train_images'
diretorio_imagens_test   = diretorio_raiz + '/test_images'
diretorio_pickles        = diretorio_raiz + '/4_modelo_classificacao'
diretorio_avaliacao      = diretorio_raiz + '/5_Avalia_Modelos'

##########################################################################
#                             DEFINO AJUSTES NAS IMAGENS                 #
##########################################################################
'''ESCOLHO PARA O TAMANHO O MAIOR TAMANHO MULTIPLO DE 2, 
QUE SEJA ABAIXO DO MINIMO'''
transf_treino     = ImageDataGenerator(rescale          = 1./255.0,
                                       validation_split = 0.5)
transf_teste      = ImageDataGenerator(rescale          = 1./255.0)
batch_size_treino = 50
batch_size_valid  = 50
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
#                             IMPORTO MODELOS                            #
##########################################################################
model_1 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch8_dropout_0.6')
model_2 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch15_1_dropout_0.6')
model_3 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch15_2_dropout_0.6')
model_4 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch16_dropout_0.6')
model_5 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch30_dropout_0.6')
model_6 = models.load_model(diretorio_pickles + '/model_cat_dog_epoch40_dropout_0.6')

##########################################################################
#                             AVALIAÇÃO - TABELA                         #
##########################################################################
VALID_PREDICT              = pd.DataFrame(columns = {'Label', 'Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4', 'Modelo 5', 'Modelo 6', 'Ensemble'})
TESTE_PREDICT              = pd.DataFrame(columns = {'Label', 'Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4', 'Modelo 5', 'Modelo 6', 'Ensemble'})
COMPARATIVO                = pd.DataFrame(columns = {'BASE', 'Metrica', 'Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4', 'Modelo 5', 'Modelo 6', 'Ensemble'})

VALID_PREDICT['Label']     = valid_generator.labels
VALID_PREDICT['Modelo 1']  = model_1.predict(valid_generator)
VALID_PREDICT['Modelo 2']  = model_2.predict(valid_generator)
VALID_PREDICT['Modelo 3']  = model_3.predict(valid_generator)
VALID_PREDICT['Modelo 4']  = model_4.predict(valid_generator)
VALID_PREDICT['Modelo 5']  = model_5.predict(valid_generator)
VALID_PREDICT['Modelo 6']  = model_6.predict(valid_generator)
VALID_PREDICT['Ensemble']  = VALID_PREDICT[['Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4', 'Modelo 5', 'Modelo 6']].max(axis = 1)

TESTE_PREDICT['Label']     = test_generator.labels
TESTE_PREDICT['Modelo 1']  = model_1.predict(test_generator)
TESTE_PREDICT['Modelo 2']  = model_2.predict(test_generator)
TESTE_PREDICT['Modelo 3']  = model_3.predict(test_generator)
TESTE_PREDICT['Modelo 4']  = model_4.predict(test_generator)
TESTE_PREDICT['Modelo 5']  = model_5.predict(test_generator)
TESTE_PREDICT['Modelo 6']  = model_6.predict(test_generator)
TESTE_PREDICT['Ensemble']  = TESTE_PREDICT[['Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4', 'Modelo 5', 'Modelo 6']].max(axis = 1)

frames = [VALID_PREDICT,
          TESTE_PREDICT]
BASE_FINAL = pd.concat(frames)
COMPARATIVO.at[0, 'BASE']     = 'Valid'
COMPARATIVO.at[0, 'Metrica']  = 'ROC'
COMPARATIVO.at[0, 'Modelo 1'] = round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 1']))*100, 2)
COMPARATIVO.at[0, 'Modelo 2'] = round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 2']))*100, 2)
COMPARATIVO.at[0, 'Modelo 3'] = round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 3']))*100, 2)
COMPARATIVO.at[0, 'Modelo 4'] = round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 4']))*100, 2)
COMPARATIVO.at[0, 'Modelo 5'] = round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 5']))*100, 2)
COMPARATIVO.at[0, 'Modelo 6'] = round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 6']))*100, 2)
COMPARATIVO.at[0, 'Modelo 6'] = round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 6']))*100, 2)
COMPARATIVO.at[0, 'Ensemble'] = round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Ensemble']))*100, 2)
COMPARATIVO.at[1, 'BASE']     = 'Teste'
COMPARATIVO.at[1, 'Metrica']  = 'ROC'
COMPARATIVO.at[1, 'Modelo 1'] = round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 1']))*100, 2)
COMPARATIVO.at[1, 'Modelo 2'] = round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 2']))*100, 2)
COMPARATIVO.at[1, 'Modelo 3'] = round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 3']))*100, 2)
COMPARATIVO.at[1, 'Modelo 4'] = round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 4']))*100, 2)
COMPARATIVO.at[1, 'Modelo 5'] = round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 5']))*100, 2)
COMPARATIVO.at[1, 'Modelo 6'] = round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 6']))*100, 2)
COMPARATIVO.at[1, 'Ensemble'] = round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Ensemble']))*100, 2)

COMPARATIVO.at[2, 'BASE']     = 'Valid'
COMPARATIVO.at[2, 'Metrica']  = 'KS'
COMPARATIVO.at[2, 'Modelo 1'] = round(ks_2samp(VALID_PREDICT.loc[VALID_PREDICT.Label == 0, 'Modelo 1'], VALID_PREDICT.loc[VALID_PREDICT.Label == 1, 'Modelo 1'])[0]*100, 2)
COMPARATIVO.at[2, 'Modelo 2'] = round(ks_2samp(VALID_PREDICT.loc[VALID_PREDICT.Label == 0, 'Modelo 2'], VALID_PREDICT.loc[VALID_PREDICT.Label == 1, 'Modelo 2'])[0]*100, 2)
COMPARATIVO.at[2, 'Modelo 3'] = round(ks_2samp(VALID_PREDICT.loc[VALID_PREDICT.Label == 0, 'Modelo 3'], VALID_PREDICT.loc[VALID_PREDICT.Label == 1, 'Modelo 3'])[0]*100, 2)
COMPARATIVO.at[2, 'Modelo 4'] = round(ks_2samp(VALID_PREDICT.loc[VALID_PREDICT.Label == 0, 'Modelo 4'], VALID_PREDICT.loc[VALID_PREDICT.Label == 1, 'Modelo 4'])[0]*100, 2)
COMPARATIVO.at[2, 'Modelo 5'] = round(ks_2samp(VALID_PREDICT.loc[VALID_PREDICT.Label == 0, 'Modelo 5'], VALID_PREDICT.loc[VALID_PREDICT.Label == 1, 'Modelo 5'])[0]*100, 2)
COMPARATIVO.at[2, 'Modelo 6'] = round(ks_2samp(VALID_PREDICT.loc[VALID_PREDICT.Label == 0, 'Modelo 6'], VALID_PREDICT.loc[VALID_PREDICT.Label == 1, 'Modelo 6'])[0]*100, 2)
COMPARATIVO.at[2, 'Ensemble'] = round(ks_2samp(VALID_PREDICT.loc[VALID_PREDICT.Label == 0, 'Ensemble'], VALID_PREDICT.loc[VALID_PREDICT.Label == 1, 'Ensemble'])[0]*100, 2)
COMPARATIVO.at[3, 'BASE']     = 'Teste'
COMPARATIVO.at[3, 'Metrica']  = 'KS'
COMPARATIVO.at[3, 'Modelo 1'] = round(ks_2samp(TESTE_PREDICT.loc[TESTE_PREDICT.Label == 0, 'Modelo 1'], TESTE_PREDICT.loc[TESTE_PREDICT.Label == 1, 'Modelo 1'])[0]*100, 2)
COMPARATIVO.at[3, 'Modelo 2'] = round(ks_2samp(TESTE_PREDICT.loc[TESTE_PREDICT.Label == 0, 'Modelo 2'], TESTE_PREDICT.loc[TESTE_PREDICT.Label == 1, 'Modelo 2'])[0]*100, 2)
COMPARATIVO.at[3, 'Modelo 3'] = round(ks_2samp(TESTE_PREDICT.loc[TESTE_PREDICT.Label == 0, 'Modelo 3'], TESTE_PREDICT.loc[TESTE_PREDICT.Label == 1, 'Modelo 3'])[0]*100, 2)
COMPARATIVO.at[3, 'Modelo 4'] = round(ks_2samp(TESTE_PREDICT.loc[TESTE_PREDICT.Label == 0, 'Modelo 4'], TESTE_PREDICT.loc[TESTE_PREDICT.Label == 1, 'Modelo 4'])[0]*100, 2)
COMPARATIVO.at[3, 'Modelo 5'] = round(ks_2samp(TESTE_PREDICT.loc[TESTE_PREDICT.Label == 0, 'Modelo 5'], TESTE_PREDICT.loc[TESTE_PREDICT.Label == 1, 'Modelo 5'])[0]*100, 2)
COMPARATIVO.at[3, 'Modelo 6'] = round(ks_2samp(TESTE_PREDICT.loc[TESTE_PREDICT.Label == 0, 'Modelo 6'], TESTE_PREDICT.loc[TESTE_PREDICT.Label == 1, 'Modelo 6'])[0]*100, 2)
COMPARATIVO.at[3, 'Ensemble'] = round(ks_2samp(TESTE_PREDICT.loc[TESTE_PREDICT.Label == 0, 'Ensemble'], TESTE_PREDICT.loc[TESTE_PREDICT.Label == 1, 'Ensemble'])[0]*100, 2)
COMPARATIVO.set_index('Metrica',
                      inplace = True)
COMPARATIVO = COMPARATIVO[['BASE', 
                           'Modelo 1', 
                           'Modelo 2', 
                           'Modelo 3', 
                           'Modelo 4', 
                           'Modelo 5', 
                           'Modelo 6',
                           'Ensemble']]
COMPARATIVO.to_csv(diretorio_avaliacao + '/comparacao_modelos_ROC_KS.csv')
BASE_FINAL.to_csv(diretorio_avaliacao + '/base_final.csv')
##########################################################################
#                             AVALIAÇÃO - GRÁFICOS                       #
##########################################################################
title_font = {'fontname' : 'Arial',
              'size'     : '17',
              'weight'   : 'bold'}
axis_font  = {'fontname' : 'Arial',
              'size'     : '12'}
lr_fpr_valid_1,  lr_tpr_valid_1,  _ = roc_curve(np.asarray(VALID_PREDICT['Label']),  np.asarray(VALID_PREDICT['Modelo 1']))
lr_fpr_valid_2,  lr_tpr_valid_2,  _ = roc_curve(np.asarray(VALID_PREDICT['Label']),  np.asarray(VALID_PREDICT['Modelo 2']))
lr_fpr_valid_3,  lr_tpr_valid_3,  _ = roc_curve(np.asarray(VALID_PREDICT['Label']),  np.asarray(VALID_PREDICT['Modelo 3']))
lr_fpr_valid_4,  lr_tpr_valid_4,  _ = roc_curve(np.asarray(VALID_PREDICT['Label']),  np.asarray(VALID_PREDICT['Modelo 4']))
lr_fpr_valid_5,  lr_tpr_valid_5,  _ = roc_curve(np.asarray(VALID_PREDICT['Label']),  np.asarray(VALID_PREDICT['Modelo 5']))
lr_fpr_valid_6,  lr_tpr_valid_6,  _ = roc_curve(np.asarray(VALID_PREDICT['Label']),  np.asarray(VALID_PREDICT['Modelo 6']))
lr_fpr_valid_E,  lr_tpr_valid_E,  _ = roc_curve(np.asarray(VALID_PREDICT['Label']),  np.asarray(VALID_PREDICT['Ensemble']))

lr_fpr_teste_1,  lr_tpr_teste_1,  _ = roc_curve(np.asarray(TESTE_PREDICT['Label']),  np.asarray(TESTE_PREDICT['Modelo 1']))
lr_fpr_teste_2,  lr_tpr_teste_2,  _ = roc_curve(np.asarray(TESTE_PREDICT['Label']),  np.asarray(TESTE_PREDICT['Modelo 2']))
lr_fpr_teste_3,  lr_tpr_teste_3,  _ = roc_curve(np.asarray(TESTE_PREDICT['Label']),  np.asarray(TESTE_PREDICT['Modelo 3']))
lr_fpr_teste_4,  lr_tpr_teste_4,  _ = roc_curve(np.asarray(TESTE_PREDICT['Label']),  np.asarray(TESTE_PREDICT['Modelo 4']))
lr_fpr_teste_5,  lr_tpr_teste_5,  _ = roc_curve(np.asarray(TESTE_PREDICT['Label']),  np.asarray(TESTE_PREDICT['Modelo 5']))
lr_fpr_teste_6,  lr_tpr_teste_6,  _ = roc_curve(np.asarray(TESTE_PREDICT['Label']),  np.asarray(TESTE_PREDICT['Modelo 6']))
lr_fpr_teste_E,  lr_tpr_teste_E,  _ = roc_curve(np.asarray(TESTE_PREDICT['Label']),  np.asarray(TESTE_PREDICT['Ensemble']))

plt.plot([0.0, 1.0], [0.0, 1.0], 'r--', linewidth = 0.5, label = 'Coin', color = 'black')
plt.plot(lr_fpr_valid_1,lr_tpr_valid_1,linewidth = 0.5, 
         label = 'Modelo 1 - ' + str(round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 1']))*100, 1)) + ' %', 
         color = 'red')
plt.plot(lr_fpr_valid_2,lr_tpr_valid_2,linewidth = 0.5, 
         label = 'Modelo 2 - ' + str(round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 2']))*100, 1)) + ' %', 
         color = 'blue')
plt.plot(lr_fpr_valid_3,lr_tpr_valid_3,linewidth = 0.5, 
         label = 'Modelo 3 - ' + str(round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 3']))*100, 1)) + ' %', 
         color = 'gold')
plt.plot(lr_fpr_valid_4,lr_tpr_valid_4,linewidth = 0.5, 
         label = 'Modelo 4 - ' + str(round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 4']))*100, 1)) + ' %', 
         color = 'darkgreen')
plt.plot(lr_fpr_valid_5,lr_tpr_valid_5,linewidth = 0.5, 
         label = 'Modelo 5 - ' + str(round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 5']))*100, 1)) + ' %', 
         color = 'darkorange')
plt.plot(lr_fpr_valid_6,lr_tpr_valid_6,linewidth = 0.5, 
         label = 'Modelo 6 - ' + str(round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 6']))*100, 1)) + ' %', 
         color = 'purple')
plt.plot(lr_fpr_valid_E,lr_tpr_valid_E,linewidth = 0.5, 
         label = 'Ensemble - ' + str(round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Ensemble']))*100, 1)) + ' %', 
         color = 'deeppink')
plt.title('Curva ROC - Validação', title_font)
plt.xlabel('False Positive Rate', axis_font)
plt.ylabel('True Positive Rate', axis_font)
plt.legend()
#plt.show()
plt.savefig(diretorio_avaliacao + '/Curvas ROC - Validação.jpg')
plt.close()

plt.plot([0.0, 1.0], [0.0, 1.0], 'r--', linewidth = 0.5, label = 'Coin', color = 'black')
plt.plot(lr_fpr_teste_1,lr_tpr_teste_1,linewidth = 0.5, 
         label = 'Modelo 1 - ' + str(round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 1']))*100, 1)) + ' %', 
         color = 'red')
plt.plot(lr_fpr_teste_2,lr_tpr_teste_2,linewidth = 0.5, 
         label = 'Modelo 2 - ' + str(round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 2']))*100, 1)) + ' %', 
         color = 'blue')
plt.plot(lr_fpr_teste_3,lr_tpr_teste_3,linewidth = 0.5, 
         label = 'Modelo 3 - ' + str(round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 3']))*100, 1)) + ' %', 
         color = 'gold')
plt.plot(lr_fpr_teste_4,lr_tpr_teste_4,linewidth = 0.5, 
         label = 'Modelo 4 - ' + str(round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 4']))*100, 1)) + ' %', 
         color = 'darkgreen')
plt.plot(lr_fpr_teste_5,lr_tpr_teste_5,linewidth = 0.5, 
         label = 'Modelo 5 - ' + str(round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 5']))*100, 1)) + ' %', 
         color = 'darkorange')
plt.plot(lr_fpr_teste_6,lr_tpr_teste_6,linewidth = 0.5, 
         label = 'Modelo 6 - ' + str(round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 6']))*100, 1)) + ' %', 
         color = 'purple')
plt.plot(lr_fpr_teste_E,lr_tpr_teste_E,linewidth = 0.5, 
         label = 'Ensemble - ' + str(round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Ensemble']))*100, 1)) + ' %', 
         color = 'deeppink')
plt.title('Curva ROC - Teste', title_font)
plt.xlabel('False Positive Rate', axis_font)
plt.ylabel('True Positive Rate', axis_font)
plt.legend()
#plt.show()
plt.savefig(diretorio_avaliacao + '/Curvas ROC - Teste.jpg')
plt.close()

plt.plot([0.0, 1.0], [0.0, 1.0], 'r--', linewidth = 0.5, label = 'Coin', color = 'black')
plt.plot(lr_fpr_valid_6,lr_tpr_valid_6,linewidth = 0.5, 
         label = 'Modelo 6 - Valid. - ' + str(round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Modelo 6']))*100, 1)) + ' %', 
         color = 'red')
plt.plot(lr_fpr_teste_6,lr_tpr_teste_6,linewidth = 0.5, 
         label = 'Modelo 6 - Teste - ' + str(round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Modelo 6']))*100, 1)) + ' %', 
         color = 'blue')
plt.title('Curva ROC - Modelo 6', title_font)
plt.xlabel('False Positive Rate', axis_font)
plt.ylabel('True Positive Rate', axis_font)
plt.legend()
#plt.show()
plt.savefig(diretorio_avaliacao + '/Curvas ROC - Modelo 6.jpg')
plt.close()

plt.plot([0.0, 1.0], [0.0, 1.0], 'r--', linewidth = 0.5, label = 'Coin', color = 'black')
plt.plot(lr_fpr_valid_E,lr_tpr_valid_E,linewidth = 0.5, 
         label = 'Ensemble - Valid. - ' + str(round(roc_auc_score(np.asarray(VALID_PREDICT['Label']), np.asarray(VALID_PREDICT['Ensemble']))*100, 1)) + ' %', 
         color = 'red')
plt.plot(lr_fpr_teste_E,lr_tpr_teste_E,linewidth = 0.5, 
         label = 'Ensemble - Teste - ' + str(round(roc_auc_score(np.asarray(TESTE_PREDICT['Label']), np.asarray(TESTE_PREDICT['Ensemble']))*100, 1)) + ' %', 
         color = 'blue')
plt.title('Curva ROC - Ensemble', title_font)
plt.xlabel('False Positive Rate', axis_font)
plt.ylabel('True Positive Rate', axis_font)
plt.legend()
#plt.show()
plt.savefig(diretorio_avaliacao + '/Curvas ROC - Ensemble.jpg')
plt.close()




