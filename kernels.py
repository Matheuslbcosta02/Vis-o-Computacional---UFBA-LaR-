import cv2
import matplotlib.pyplot as plt
import numpy as np

imagem_original = cv2.imread("kernel_imagem_bancada.jpg", cv2.IMREAD_GRAYSCALE)

altura, largura = imagem_original.shape

total_pixels_imagem = altura * largura

menor_intensidade_img_original = np.min(imagem_original)
maior_intensidade_img_original = np.max(imagem_original)
media_intensidade_img_original = np.mean(imagem_original)

print(f'Largura {largura}, altura {altura}, total pixels {total_pixels_imagem}\n\n')
print(f'Intensidades:\n')
print(f'Min: {menor_intensidade_img_original}, Max: {maior_intensidade_img_original}, Média: {media_intensidade_img_original}')

plt.imshow(imagem_original, cmap='gray')
plt.show()


kernel_media = np.ones(shape=(3,3)) * 1/9

print(kernel_media)

print(np.sum(kernel_media))

imagem_filtrada_kernel_media = cv2.filter2D(imagem_original,-1,kernel_media)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(imagem_original, cmap='gray')
plt.title('Imagem Original')
plt.subplot(1,2,2)
plt.imshow(imagem_filtrada_kernel_media, cmap='gray')
plt.title('Imagem Filtrada - Kernel Média')
plt.show()

# construir kernel gaussiano istropico sigma_u = sigma_v

sigma = 1
raio = int(np.ceil(3* sigma))
#vetores coordenadas espaciais -> posicoes discretas dos pixels em relação ao centro zero zero do kernel -3 ate 3
u=np.arange(-raio, raio+1)
v = np.arange(-raio,raio+1)

#mapa da posicao de cada pixel fica no espaço da janela
U, V = np.meshgrid(u,v) #gerando duas matrizes 2d U coordenadas horizontal de cada celula da janela e V vertical

# não precisa coolocar dividir por 2pi sigma^2 pois vou normalizar depois dividindo tudo por sum G

matriz_G =  np.exp(-(U**2 + V**2)/2)

Kernel_Gaussiano = matriz_G/np.sum(matriz_G)

print(Kernel_Gaussiano)

print(Kernel_Gaussiano.shape)

print(f'Soma dos coeficientes após a normalização: {np.sum(Kernel_Gaussiano)}')

#represetar grafica bidimensional dos pesos
plt.title('Bidimensional do pesos')
plt.imshow(Kernel_Gaussiano, cmap='viridis')
plt.show()

#representancao tridimensional da superficie gaussiana
ax=plt.subplot(projection='3d') #subplot 3d
plt.title('Tridimensional superficie Gaussiana')
ax.plot_surface(U,V,Kernel_Gaussiano,cmap='viridis')
plt.show()

imagem_filtrada_kernel_gaussiano = cv2.filter2D(imagem_original,-1,Kernel_Gaussiano)

plt.title("Imagem filtrada - Kernel Gaussiano")
plt.imshow(imagem_filtrada_kernel_gaussiano, cmap='gray')
plt.show()


sigma1 = 0.8
sigma2 = 1.5
sigma3 = 3.0

raio2 = int(np.ceil(sigma2 * 3))
raio1 = int(np.ceil(sigma1 * 3))
raio3= int(np.ceil(sigma3 * 3))

u1 = np.arange(-raio1, raio1 +1)
v1 = np.arange(-raio1, raio1+1)

u2 = np.arange(-raio2, raio2 +1)
v2 = np.arange(-raio2, raio2+1)

u3 = np.arange(-raio3, raio3 +1)
v3 = np.arange(-raio3, raio3+1)

U1, V1 = np.meshgrid(u1,v1)
U2, V2 = np.meshgrid(u2,v2)
U3, V3 = np.meshgrid(u3,v3)

matriz_G1 =  np.exp(-(U1**2/sigma1**2 + V1**2/sigma1**2)/2)

Kernel_Gaussiano1 = matriz_G1/np.sum(matriz_G1)


matriz_G2 =  np.exp(-(U2**2/sigma2**2 + V2**2/sigma2**2)/2)

Kernel_Gaussiano2 = matriz_G2/np.sum(matriz_G2)

matriz_G3 =  np.exp(-(U3**2/sigma3**2 + V3**2/sigma3**2)/2)

Kernel_Gaussiano3 = matriz_G3/np.sum(matriz_G3)

imagem_filtrada_kernel_gaussiano1 = cv2.filter2D(imagem_original,-1,Kernel_Gaussiano1)
imagem_filtrada_kernel_gaussiano2 = cv2.filter2D(imagem_original,-1,Kernel_Gaussiano2)
imagem_filtrada_kernel_gaussiano3 = cv2.filter2D(imagem_original,-1,Kernel_Gaussiano3)

plt.figure(figsize=(18,10))
plt.subplot(1,4,1)
plt.title('Imagem original')
plt.imshow(imagem_original, cmap='gray')
plt.subplot(1,4,2)
plt.title(f'Imagem filtrada com sigma = {sigma1}')
plt.imshow(imagem_filtrada_kernel_gaussiano1, cmap='gray')
plt.subplot(1,4,3)
plt.title(f'Imagem filtrada com sigma = {sigma2}')
plt.imshow(imagem_filtrada_kernel_gaussiano2, cmap='gray')
plt.subplot(1,4,4)
plt.title(f'Imagem filtrada com sigma = {sigma3}')
plt.imshow(imagem_filtrada_kernel_gaussiano3, cmap='gray')
plt.show()

## meu kernel gaussiano inicial era 7x7 vou fazer um de media 7x7 tmb
kernel_media7por7 = np.ones(shape=(7,7)) * 1/49
Kernel_Gaussiano7por7 = matriz_G/np.sum(matriz_G)

imagem_filtrada_kernel_media_7por_7 = cv2.filter2D(imagem_original, -1, kernel_media7por7)
imagem_filtrada_gaussiano_7por7 = cv2.filter2D(imagem_original,-1,Kernel_Gaussiano7por7)
plt.figure(figsize=(18,10))
plt.subplot(1,3,1)
plt.title('Imagem original')
plt.imshow(imagem_original, cmap='gray')
plt.subplot(1,3,2)
plt.title('Imagem filtrada com filtro de média 7x7')
plt.imshow(imagem_filtrada_kernel_media_7por_7, cmap='gray')
plt.subplot(1,3,3)
plt.title('Imagem filtrada com filtro Gaussiano 7x7')
plt.imshow(imagem_filtrada_gaussiano_7por7, cmap='gray')
plt.show()

#agora kernel gaussiano não isotropico

n_sigma_u1 = 3
n_sigma_v1 = 0.8

n_raio_u1 = int(np.ceil(n_sigma_u1 * 3))
n_raio_v1 = int(np.ceil(n_sigma_v1 * 3))

n_u1 = np.arange(-n_raio_u1, n_raio_u1+1 )
n_v1 = np.arange(-n_raio_v1, n_raio_v1+1)


nU_1,nV_1 = np.meshgrid(n_u1, n_v1)

matriz_n_G1 =  np.exp(-(nU_1**2/n_sigma_u1**2 + nV_1**2/n_sigma_v1**2)/2)

Kernel_Gaussiano_n_1 = matriz_n_G1/np.sum(matriz_n_G1)

imagem_filtrada_kernel_nao_iso_1 = cv2.filter2D(imagem_original,-1,Kernel_Gaussiano_n_1)

n_sigma_u2 = 0.8
n_sigma_v2 = 3

n_raio_u2 = int(np.ceil(n_sigma_u2 * 3))
n_raio_v2 = int(np.ceil(n_sigma_v2 * 3))

n_u2 = np.arange(-n_raio_u2, n_raio_u2+1 )
n_v2 = np.arange(-n_raio_v2, n_raio_v2+1)


nU_2,nV_2 = np.meshgrid(n_u2, n_v2)

matriz_n_G2 =  np.exp(-(nU_2**2/n_sigma_u2**2 + nV_2**2/n_sigma_v2**2)/2)

Kernel_Gaussiano_n_2 = matriz_n_G2/np.sum(matriz_n_G2)

imagem_filtrada_kernel_nao_iso_2 = cv2.filter2D(imagem_original,-1,Kernel_Gaussiano_n_2)

print(f'dimensão da gaussiana não isotropica 1: {Kernel_Gaussiano_n_1.shape}')
print(f'dimensão da gaussiana não isotropica 2: {Kernel_Gaussiano_n_2.shape}')

plt.subplot(1,2,1)
plt.title('Bidimensional do pesos Kernel Gaussiano não isotropico 1')
plt.imshow(Kernel_Gaussiano_n_1, cmap='viridis')
plt.subplot(1,2,2)
plt.title('Bidimensional do pesos Kernel Gaussiano não isotropico 2')
plt.imshow(Kernel_Gaussiano_n_2, cmap='viridis')
plt.show()


ax=plt.subplot(1,2,1,projection='3d') #subplot 3d
plt.title('Tridimensional superficie Kernel Gaussiano não isotorpico 1')
ax.plot_surface(nU_1,nV_1,Kernel_Gaussiano_n_1,cmap='viridis')
ax.set_xlabel('U')
ax.set_ylabel('V')
ax.set_zlabel('Peso')
ax2 = plt.subplot(1,2,2, projection='3d')
plt.title('Tridimensional superficie Kernel Gaussiano não isotorpico 2')
ax2.plot_surface(nU_2,nV_2,Kernel_Gaussiano_n_2,cmap='viridis')
ax2.set_xlabel('U')
ax2.set_ylabel('V')
ax2.set_zlabel('Peso')
plt.show()



plt.figure(figsize=(20,10))
plt.subplot(2,3,1)
plt.title('Imagem original')
plt.imshow(imagem_original, cmap='gray')
plt.subplot(2,3,3)
plt.title('Filtragem Gaussiana Isotropica')
plt.imshow(imagem_filtrada_gaussiano_7por7, cmap='gray')
plt.subplot(2,3,4)
plt.title('Imagem com gaussiana não isotropica sigma_u = 3 e sigma_v = 0.8')
plt.imshow(imagem_filtrada_kernel_nao_iso_1, cmap='gray')
plt.subplot(2,3,6)
plt.title('Imagem com gaussiana não isotropica sigma_u = 0.8 e sigma_v = 3')
plt.imshow(imagem_filtrada_kernel_nao_iso_2, cmap='gray')
plt.subplots_adjust(hspace=0.5)
plt.show()

altura_do_iso, largura_do_iso = Kernel_Gaussiano7por7.shape
centro_do_iso_7por7 = altura_do_iso//2
perfil_horizontal_kernel_isotropico = Kernel_Gaussiano7por7[centro_do_iso_7por7,:]
perfil_vertical_kernel_isotropico = Kernel_Gaussiano7por7[:,centro_do_iso_7por7]

altura_nao_iso_1 , largura_nao_iso_1 = Kernel_Gaussiano_n_1.shape

altura_nao_iso_2 , largura_nao_iso_2 = Kernel_Gaussiano_n_2.shape

print(altura_nao_iso_1, largura_nao_iso_1)
print(altura_nao_iso_2, largura_nao_iso_2)


centro_linha_nao_iso1 = altura_nao_iso_1//2
centro_coluna_nao_iso1 = largura_nao_iso_1//2
perfil_horizontal_kernel_nao_iso_1 = Kernel_Gaussiano_n_1[centro_linha_nao_iso1,:]
perfil_vertical_kernel_nao_iso_1 = Kernel_Gaussiano_n_1[:, centro_coluna_nao_iso1]

centro_linha_nao_iso2 = altura_nao_iso_2//2
centro_coluna_nao_iso2 = largura_nao_iso_2//2
perfil_horizontal_kernel_nao_iso_2 = Kernel_Gaussiano_n_2[centro_linha_nao_iso2,:]
perfil_vertical_kernel_nao_iso_2 = Kernel_Gaussiano_n_2[:, centro_coluna_nao_iso2]

plt.figure(figsize=(18,9))
plt.suptitle('Perfil Horizontal dos Kernels')
plt.subplot(1,3,1)
plt.title('Kernel Isotrópico')
plt.plot(perfil_horizontal_kernel_isotropico)
plt.subplot(1,3,2)
plt.title('Kernel não iso 1')
plt.plot(perfil_horizontal_kernel_nao_iso_1)
plt.subplot(1,3,3)
plt.title('kernel não iso 2')
plt.plot(perfil_horizontal_kernel_nao_iso_2)
plt.show()

#formato da curva -> valor do desvio padrao na direção

plt.figure(figsize=(18,9))
plt.suptitle('Perfil Vertical dos Kernels')
plt.subplot(1,3,1)
plt.title('Kernel Isotrópico')
plt.plot(perfil_vertical_kernel_isotropico)
plt.subplot(1,3,2)
plt.title('Kernel não iso 1')
plt.plot(perfil_vertical_kernel_nao_iso_1)
plt.subplot(1,3,3)
plt.title('kernel não iso 2')
plt.plot(perfil_vertical_kernel_nao_iso_2)
plt.show()
