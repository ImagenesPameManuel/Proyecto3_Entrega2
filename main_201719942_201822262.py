#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto3 Entrega2
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
import random
import numpy as np
from skimage.filters import threshold_otsu
import skimage.io as io
import skimage.morphology as morfo
import skimage.segmentation as segmen
from scipy.ndimage import binary_dilation
import sklearn.metrics as skmetr
from scipy.stats import mode
import matplotlib.pyplot as plt
import os
import cv2
import glob
import pandas
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
img = glob.glob(os.path.join('data_mp3','blood_cell_dataset', 'noisy_data', '*.png')) # se importan las imágenes
preprocesadas=[] # lista para almacenar imágenes
for i in img:
    carga_color_image=io.imread(i)
    imagen_en_Lab = cv2.cvtColor(carga_color_image, cv2.COLOR_BGR2LAB)  # se convierte de rgb a Lab con librería cv2
    filtrado= cv2.medianBlur(imagen_en_Lab,7)
    espacio_L = filtrado[:, :, 0]  # se extraen canales de la imagen en el espacio de color La*b
    espacio_a = filtrado[:, :, 1]
    espacio_b = filtrado[:, :, 2]
    umbral = threshold_otsu(espacio_b) #cálculo del umbral de otsu y binarización de la imagen
    binarizada = espacio_b < umbral
    preprocesadas.append(binarizada) #se añade nueva imagen preprocesada a arreglo
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
def ImgComplemento(img): # funcion para calcular el complemento
    complemento = img.copy() # se copia la imagen ingresada por parametro
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 0:
                complemento[i][j] = 1 # se intercambian los valores de 1 con 0
            else:
                complemento[i][j] = 0 # se intercambian los valores de 0 con 1
    return complemento
def MyHoleFiller_201719942_201822262(bin_img):
    complemento = ImgComplemento(bin_img) # se saca el complemento de la imagen ingresada por parámetro
    conbordecomplemento = np.zeros((len(complemento), len(complemento[0]))) # se crea matriz de ceros
    for i in range(len(bin_img)):
        for j in range(len(bin_img[0])):
            if i == 0 or i == len(complemento) - 1 or j == 0 or j == len(complemento[0]) - 1:
                conbordecomplemento[i][j] = complemento[i][j] # se llenan los bordes de la matriz con el complemento de la imegen original
    marcador_inicialF = conbordecomplemento # variable para el marcador
    mask = complemento # variable para la máscara
    elemt_struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) # elemento estructurante seleccionado arbitrariamente
    marcador = marcador_inicialF
    fin = False
    while fin == False:
        previo = np.copy(marcador) # se copia el marcador anterior
        marcador = binary_dilation(marcador, elemt_struct).astype(marcador.dtype) # se realiza una dilatación del marcador con el elemento estructurante
        marcador = marcador + mask # se hace una suma de las matrices para revisar la intersección
        for i in range(len(marcador)):
            for j in range(len(marcador[0])):
                if marcador[i][j] > 1: # dónde haya interseccion se preserva con un 1
                    marcador[i][j] = 1
                else:
                    marcador[i][j] = 0 # dónde no hay intersección se convierte en fondo con un 0
        if np.array_equal(previo, marcador):
            fin = True # condición de parada: si el marcador anterior es igual al actual
    whole_img=ImgComplemento(marcador)
    return whole_img # se retorna el resultado
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
def MyConnComp_201719942_201822262(binary_image, conn = 4):
    elemt_struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # elemento estructurante para conectividad default 4
    if conn == 8:
        elemt_struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # elemento estructurante para conectividad 8
    mask = np.copy(binary_image)
    labeled_image = np.zeros((len(binary_image), len(binary_image[0])))
    tatuaje = 1 # con lo que se va a marcar cada elemento del componenete conexo
    for i in range(len(binary_image)): # se hace dilatación geodésica
        for j in range(len(binary_image[0])):
            if mask[i][j] == 1:
                papel_cal = np.zeros((len(binary_image), len(binary_image[0])))
                papel_cal[i][j] = mask[i][j]
                fin = False
                while fin == False:
                    previo = np.copy(papel_cal)  # se copia el marcador anterior
                    papel_cal = binary_dilation(papel_cal, elemt_struct).astype(papel_cal.dtype)  # se realiza una dilatación del marcador con el elemento estructurante
                    papel_cal = papel_cal + mask  # se hace una suma de las matrices para revisar la intersección
                    for ii in range(len(papel_cal)):
                        for jj in range(len(papel_cal[0])):
                            if papel_cal[ii][jj] > 1:  # dónde haya interseccion se preserva con un 1
                                papel_cal[ii][jj] = 1
                            else:
                                papel_cal[ii][jj] = 0  # dónde no hay intersección se convierte en fondo con un 0
                    if np.array_equal(previo, papel_cal):
                        fin = True  # condición de parada: si el marcador anterior es igual al actual
                mask = mask - papel_cal # se quita el componente conexo
                labeled_image += tatuaje * papel_cal # se modifica la matriz donde se guardan los componentes conexos
                tatuaje += 1 # se aumenta el identificador de componente conexo
    pixel_labels = np.array([])
    for t in range(1, tatuaje): # para todos los componentes conexos identificados se va a extraer los índices
        xces = []
        yes = []
        for z in range(len(labeled_image)):
            for j in range(len(labeled_image[0])):
                if t == labeled_image[z][j]:
                    xces.append(z)
                    yes.append(j)
        pixel_labels = np.append(pixel_labels, np.ravel_multi_index((xces, yes), (len(binary_image), len(binary_image[0])))) # se extraen los índices correspondientes al componente conexo
    return labeled_image, pixel_labels
prueba_frutas=io.imread(os.path.join('data_mp3', 'fruits_binary.png')) # se importa la imagen de prueba
# se binariza la imagen cambiando valores de 255 por 1
prueba_frutas=prueba_frutas>0
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure() # se plotea la imagen de las frutas sin utilizar el algoritmo de componentes conexos y con el algoritmo
plt.subplot(1,2,1)
plt.title("Imagen original")
plt.axis("off")
plt.imshow(prueba_frutas,cmap="gray")
plt.subplot(1,2,2)
plt.title("Imagen componentes conexos")
plt.axis("off")
plt.imshow(MyConnComp_201719942_201822262(prueba_frutas)[0],cmap="gray")
plt.show()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
# se crean 2 imágenes binarias 20 X 20
imagen_creada1=np.array([[0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],[1,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0],[1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1],[0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0],[0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0]])
imagen_creada2=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1],[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1],[0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1],[0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0],[0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0]])
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure("vecindad_4_8_Dif") # se aplica el algoritmo para la primera imagen con dos vecindades diferentes y se muestran. se muestra también la imagen original
plt.subplot(1,3,1)
plt.title("Imagen original")
plt.axis("off")
plt.imshow(imagen_creada1,cmap="gray")
plt.subplot(1,3,2)
plt.title("Imagen componentes\nconexos con 4 vecindad")
plt.axis("off")
plt.imshow(MyConnComp_201719942_201822262(imagen_creada1)[0],cmap="gray")
plt.subplot(1,3,3)
plt.title("Imagen componentes\nconexos con 8 vecindad")
plt.axis("off")
plt.imshow(MyConnComp_201719942_201822262(imagen_creada1,conn=8)[0],cmap="gray")
plt.show()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure("vecindad_4_8_Igual") # se aplica el algoritmo para la primera imagen con dos vecindades diferentes y se muestran. se muestra también la imagen original
plt.subplot(1,3,1)
plt.title("Imagen original")
plt.axis("off")
plt.imshow(imagen_creada2,cmap="gray")
plt.subplot(1,3,2)
plt.title("Imagen componentes\nconexos con 4 vecindad")
plt.axis("off")
plt.imshow(MyConnComp_201719942_201822262(imagen_creada2)[0],cmap="gray")
plt.subplot(1,3,3)
plt.title("Imagen componentes\nconexos con 8 vecindad")
plt.axis("off")
plt.imshow(MyConnComp_201719942_201822262(imagen_creada2,conn=8)[0],cmap="gray")
plt.show()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
img = glob.glob(os.path.join('data_mp3','blood_cell_dataset', 'noisy_data', '*.png')) # se importan las imágenes
#num_rand=random.randint(0,9) #proceso para hallar imagen de prueba de forma aleatoria
#print(num_rand) #=7
carga_prueba=io.imread(img[7]) # imagen para realizar las pruebas
def preprocesamiento(carga_color_image): # función para preprocesar las imágenes
    imagen_en_Lab = cv2.cvtColor(carga_color_image, cv2.COLOR_BGR2LAB)  # se convierte de rgb a Lab con librería cv2
    filtrado = cv2.medianBlur(imagen_en_Lab, 7) #se filtra la imagen
    filtrado_grises = cv2.cvtColor(cv2.cvtColor(filtrado, cv2.COLOR_LAB2RGB),cv2.COLOR_RGB2GRAY)  # se convierte de rgb a Lab con librería cv2
    return filtrado_grises
def gradiente_morfo(image): # función para calcular el gradiente morfologico
    dilatacion=morfo.dilation(image)
    erosion=morfo.erosion(image)
    return dilatacion-erosion # con librería se calculan los valores y se retorna la resta
def watershed_select(image,marcadores=False,min_h=40): # función para calcular watershed
    grad=gradiente_morfo(image) # se saca el gradiente morfologico
    if marcadores==True: # si se va a calcular con marcadores se utiliza la funcion h_minima
        marks=morfo.h_minima(image,min_h)
        conexos=MyConnComp_201719942_201822262(marks)[0] # se sacan componentes conexos
        return segmen.watershed(grad,markers=conexos,watershed_line=True) ,conexos # se hace watershed
    else:
        return segmen.watershed(grad,watershed_line=True) # por default se hace watershed sin marcadores
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure() # se plotean las imágenes resultantes
plt.subplot(1,3,1)
plt.title("Imagen original preprocesada")
plt.axis("off")
plt.imshow(preprocesamiento(carga_prueba),cmap="gray")
plt.subplot(1,3,2)
plt.title("Gradiente morfológico")
plt.axis("off")
plt.imshow(gradiente_morfo(preprocesamiento(carga_prueba)),cmap="gray")
plt.subplot(1,3,3)
plt.title("Watershed sin\nmardadores definidos")
plt.axis("off")
plt.imshow(watershed_select(preprocesamiento(carga_prueba)),cmap="gray")
plt.show()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure() # se plotean las imágenes resultantes
plt.subplot(2,2,1)
plt.title("Imagen original preprocesada")
plt.axis("off")
plt.imshow(preprocesamiento(carga_prueba),cmap="gray")
plt.subplot(2,2,2)
plt.title("Gradiente morfológico")
plt.axis("off")
plt.imshow(gradiente_morfo(preprocesamiento(carga_prueba)),cmap="gray")
plt.subplot(2,2,3)
plt.title("Mardadores definidos")
plt.axis("off")
plt.imshow(watershed_select(preprocesamiento(carga_prueba),marcadores=True)[1],cmap="gray")
plt.show()
plt.subplot(2,2,4)
plt.title("Watershed\nmardadores definidos")
plt.axis("off")
plt.imshow(watershed_select(preprocesamiento(carga_prueba),marcadores=True,min_h=255-threshold_otsu(preprocesamiento(carga_prueba)))[0] ,cmap="gray")
plt.show()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#carga anotaciones y cálculo de índices de Jaccard para método entrega 1
img_anot = glob.glob(os.path.join('data_mp3','blood_cell_dataset', 'groundtruth', '*.png')) # lectura de imagenes
anota,jaccards_llenas=[],{}# diccionario para guardar las métricas y lista para guardar anotaciones
for i in img_anot:
    carga=io.imread(i)>0
    anota.append(carga) # se realiza la carga y umbralización de imagenes
for i in range(len(preprocesadas)):
    llena=MyHoleFiller_201719942_201822262(preprocesadas[i]).flatten() # se rellenan los huecos
    i_dict,i_anota=i,i #indices correspondientes al número de la imagen
    if i == 1: # condicionales para excepciones en el orden correspondiente
        i_dict=10
        i_anota=0
    elif i==0:
        i_dict = i+1
        i_anota=1
    jaccards_llenas[i_dict] = skmetr.jaccard_score(anota[i_anota].flatten(), llena)#cálculo de métricas y asignación de valores en los diccionarios haciendo uso de las anotaciones e imagenes correspondientes
# se muestran las métricas
valores_llenos=np.array(list(jaccards_llenas.values()))
print("\nÍndices de Jaccard imágenes sin huecos:\n",jaccards_llenas)
print("prom llenos",np.mean(valores_llenos),"desv.est",np.std(valores_llenos))
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
segm_water={} # lista para almacenar imágenes
jaccards_sinmark={}
jaccards_mark={}
index_im = 0
for i in img: # se recorren las imagenes
    carga_color=io.imread(i) # se leen
    preprocesa=preprocesamiento(carga_color)  # se preprocesan
    segm_sinmarcadores=watershed_select(preprocesa)>0 # se hace watershed sin marcadores
    segm_marcadores=watershed_select(preprocesa,marcadores=True,min_h=255-threshold_otsu(preprocesa))[0] # se hace watershed son marcadores con umbral especificado
    segm_marcadores_bin = segm_marcadores.copy() # se copia el resultado
    moda = mode(segm_marcadores_bin.flatten())[0] # se obtiene el valor del fondo a través de la moda
    for i in range(len(segm_marcadores_bin)): # se recorre la imagen para realizar una binarización
        for j in range(len(segm_marcadores_bin[0])): # si no es fondo se convierte en 1
            if segm_marcadores_bin[i][j] < moda or segm_marcadores_bin[i][j] > moda:
                segm_marcadores_bin[i][j] = 1
            else: # si es fondo se convierte en 0
                segm_marcadores_bin[i][j] = 0
    #plt.imshow(segm_marcadores_bin, cmap="gray")
    i_dict, i_anota = index_im, index_im  # indices correspondientes al número de la imagen
    if index_im == 1:  # condicionales para excepciones en el orden correspondiente
        i_dict = 10
        i_anota = 0
    elif index_im == 0:
        i_dict = index_im+ 1
        i_anota = 1
    segm_water[i_dict]=[preprocesa,segm_sinmarcadores,segm_marcadores_bin]
    jaccards_sinmark[i_dict] = skmetr.jaccard_score(anota[i_anota].flatten(), segm_sinmarcadores.flatten())  # cálculo de métricas y asignación de valores en los diccionarios haciendo uso de las anotaciones e imagenes correspondientes
    jaccards_mark[i_dict] = skmetr.jaccard_score(anota[i_anota].flatten(), segm_marcadores_bin.flatten())
    index_im+=1 # aumento del índice
valores_sinmark=np.array(list(jaccards_sinmark.values())) #creacion de lista para calcular métricas
valores_mark=np.array(list(jaccards_mark.values())) #creacion de lista para calcular métricas
# se muestran las métricas
print("\nÍndices de Jaccard imágenes watersheds sin marcadores:\n",jaccards_sinmark)
print("prom sin marcadores",np.mean(valores_sinmark),"desv.est",np.std(valores_sinmark))
print("\nÍndices de Jaccard imágenes con marcadores:\n",jaccards_mark)
print("prom con marcadores",np.mean(valores_mark),"desv.est",np.std(valores_mark))