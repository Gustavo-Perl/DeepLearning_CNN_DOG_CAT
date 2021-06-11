##########################################################################
#                             IMPORTO LIBRARIES                          #
##########################################################################
import numpy      as np
import pandas     as pd
import glob
import os
from pathlib import Path

##########################################################################
#                             DEFINO DIRETORIOS                          #
##########################################################################
diretorio_raiz           = 'C:/Users/gusta/Desktop/Deep Learning'
diretorio_imagens_treino = diretorio_raiz + '/train_images'
diretorio_imagens_teste  = diretorio_raiz + '/test_images'
diretorio_modelo         = diretorio_raiz + '/4_modelo_classificacao'

##########################################################################
#                             INICIO MUDANÇA DE DIRETORIO                #
##########################################################################
base_depara_id = pd.DataFrame(pd.read_csv(diretorio_modelo + '/3_de_para_json_teste_originals.csv',
                                          encoding = 'utf8',
                                          sep = ';',
                                          error_bad_lines = True))
for i in range(0, base_depara_id.shape[0]):
    if base_depara_id['Tipo'][i] == 'cat':
        Path(diretorio_imagens_teste + '/' + base_depara_id['ID'][i] + '.jpg').rename(diretorio_imagens_teste + '/cat/' + base_depara_id['ID'][i] + '.jpg')
    else:
        Path(diretorio_imagens_teste + '/' + base_depara_id['ID'][i] + '.jpg').rename(diretorio_imagens_teste + '/dog/' + base_depara_id['ID'][i] + '.jpg')
 