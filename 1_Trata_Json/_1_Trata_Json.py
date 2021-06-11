##########################################################################
#                             IMPORTO LIBRARIES                          #
##########################################################################
import numpy      as np
import pandas     as pd
import random     as rd
import json
from os import listdir
from os.path import isfile, join
##########################################################################
#                             DEFINO CAMINHOS                            #
##########################################################################
diretorio_raiz             = 'C:/Users/gusta/Desktop/Deep Learning'
diretorio_codigo           = diretorio_raiz + '/1_Trata_Json'
diretorio_depara_id_treino = diretorio_raiz + '/train_metadata'
diretorio_depara_id_teste  = diretorio_raiz + '/test_metadata'
diretorio_modelo           = diretorio_raiz + '/4_modelo_classificacao'

##########################################################################
#                             IMPORTO JSON - TREINO                      #
##########################################################################
onlyfiles  = listdir(diretorio_depara_id_treino)
ID_base    = []
Tipo_base  = []
for x in range(len(onlyfiles)):
    try: 
        Tipo_base.append(json.load(open(diretorio_depara_id_treino + '/' + onlyfiles[x], 
                                   encoding = 'utf8'))['labelAnnotations'][0]['description'])
        ID_base.append(onlyfiles[x].split('.json')[0])
    except Exception:
        pass

##########################################################################
#                             CRIO BASE UNICA - TREINO                   #
##########################################################################
final_base = pd.DataFrame(columns = {'ID', 'Tipo'})
final_base['ID'] = ID_base
final_base['Tipo'] = Tipo_base
final_base.set_index('Tipo',
                     inplace = True)
final_base = final_base.loc[['cat','dog','dog breed','dog like mammal']]
final_base.reset_index(inplace = True)
final_base['Tipo'] = np.where(final_base['Tipo'] == 'dog like mammal',
                              'dog',
                     np.where(final_base['Tipo'] == 'dog breed',
                              'dog', final_base['Tipo']))
final_base.set_index('ID',
                     inplace = True)
final_base.to_csv(diretorio_modelo + '/de_para_json_treino.csv',
                  sep = ';',
                  encoding = 'utf8')

##########################################################################
#                             IMPORTO JSON - TESTE                       #
##########################################################################
onlyfiles  = listdir(diretorio_depara_id_teste)
ID_base    = []
Tipo_base  = []
for x in range(len(onlyfiles)):
    try: 
        Tipo_base.append(json.load(open(diretorio_depara_id_teste + '/' + onlyfiles[x], 
                                   encoding = 'utf8'))['labelAnnotations'][0]['description'])
        ID_base.append(onlyfiles[x].split('.json')[0])
    except Exception:
        pass

##########################################################################
#                             CRIO BASE UNICA - TESTE                    #
##########################################################################
final_base = pd.DataFrame(columns = {'ID', 'Tipo'})
final_base['ID'] = ID_base
final_base['Tipo'] = Tipo_base
final_base.set_index('Tipo',
                     inplace = True)
final_base = final_base.loc[['cat','dog','dog breed','dog like mammal']]
final_base.reset_index(inplace = True)
final_base['Tipo'] = np.where(final_base['Tipo'] == 'dog like mammal',
                              'dog',
                     np.where(final_base['Tipo'] == 'dog breed',
                              'dog', final_base['Tipo']))
final_base.set_index('ID',
                     inplace = True)
final_base.to_csv(diretorio_modelo + '/de_para_json_teste.csv',
                  sep = ';',
                  encoding = 'utf8')