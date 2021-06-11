##########################################################################
#                             IMPORTO LIBRARIES                          #
##########################################################################
import numpy      as np
import pandas     as pd
import glob
import os
from random                    import randint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

##########################################################################
#                             DEFINO DIRETORIOS                          #
##########################################################################
diretorio_raiz          = 'C:/Users/gusta/Desktop/Deep Learning'
diretorio_codigo        = diretorio_raiz + '/2_Cria_Imagens'
diretorio_imagens       = diretorio_raiz + '/test_images'
diretorio_novas_imagens = diretorio_raiz + '/new_test_images'
diretorio_modelo        = diretorio_raiz + '/4_modelo_classificacao'

##########################################################################
#                             INICIO CRIAÇÃO DE NOVAS IMAGENS            #
##########################################################################
base_depara_id = pd.DataFrame(pd.read_csv(diretorio_modelo + '/de_para_json_teste.csv',
                                          encoding = 'utf8',
                                          sep = ';',
                                          error_bad_lines = True)).set_index('ID')
base_depara_id = base_depara_id.sample(frac = 1).reset_index()
de_para_new_images = pd.DataFrame(columns = {'ID',
                                             'Tipo',
                                             'Width',
                                             'Height'})
ID_base       = []
Tipo_base     = []
Height_base   = []
Width_base    = []
transformacao = ImageDataGenerator(rotation_range     = 30,        #rotaciona
                                    width_shift_range  = 0.2,       #aumenta largura
                                    height_shift_range = 0.2,       #aumenta altura
                                    zoom_range         = [0.4,1.0], #zoom
                                    brightness_range   = [0.2,1.0], #nitidez
                                    horizontal_flip    = True,      #inverte na horizontal
                                    vertical_flip      = True,      #inverte na vertical
                                    shear_range        = 0.2,       #distorções
                                    fill_mode          = 'constant')

for i in range(0, base_depara_id.shape[0]):
    imagem = load_img(diretorio_imagens + '/' + base_depara_id['ID'][i] + '.jpg',
                      color_mode = 'rgb',
                      grayscale = False)
    array = img_to_array(imagem,
                         data_format = 'channels_last')
    array = array.reshape((1,) + array.shape)

    var_name  = randint(2000000, 3000000)
    ID_base.append(str(var_name))
    Tipo_base.append(base_depara_id['Tipo'][i])
    Width_base.append(imagem.size[0])
    Height_base.append(imagem.size[1])
    for batch in transformacao.flow(array, 
                                    batch_size  = 1,
                                    save_to_dir = diretorio_novas_imagens,
                                    save_prefix = str(var_name),
                                    save_format = 'jpg'):
        break

base_depara_id['Width']  = Width_base
base_depara_id['Height'] = Height_base
de_para_new_images['ID'] = ID_base
de_para_new_images['Tipo'] = Tipo_base
de_para_new_images['Width'] = Width_base
de_para_new_images['Height'] = Height_base
base_depara_id.set_index('ID',
                         inplace = True)
base_depara_id.to_csv(diretorio_modelo + '/3_de_para_json_teste_originals.csv',
                      sep = ';',
                      encoding = 'utf8')
de_para_new_images.set_index('ID',
                             inplace = True)
de_para_new_images.to_csv(diretorio_modelo + '/4_de_para_json_teste_new.csv',
                          sep = ';',
                          encoding = 'utf8')