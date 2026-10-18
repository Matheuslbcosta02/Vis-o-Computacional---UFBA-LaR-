import cv2
import numpy as np
import matplotlib.pyplot as plt

imagem_original = cv2.imread('imagem_original_bordas.png', cv2.IMREAD_GRAYSCALE)

plt.imshow(imagem_original, cmap='gray')
plt.show()

altura, largura = imagem_original.shape
numero_total_pixels = altura * largura
menor_intensidade = np.min(imagem_original)
maior_intensidade = np.max(imagem_original)
intensidade_media = np.mean(imagem_original)

print(f'Altura: {altura}, largura: {largura}, total de pixels: {numero_total_pixels}')
print()
print(f'menor intensidade na imagem: {menor_intensidade}, maior intensidade: {maior_intensidade}, intensidade média: {intensidade_media}')
print()

matriz = [
    [-1,0,1],[-2,0,2],[-1,0,1]
]

kv = np.array(matriz, dtype=np.float32)
ku = kv.T
print(f'kv: {kv}')
print()
print(f'ku: {ku}')

imagem_float = imagem_original.astype(np.float32) #conveter pra não truncar em zero quando derivadas negativas


Imagem_v = cv2.filter2D(imagem_float,cv2.CV_32F, kv)
Imagem_u = cv2.filter2D(imagem_float,cv2.CV_32F, ku)

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.title('kv - bordas horizontais')
plt.imshow(Imagem_v, cmap='gray')
plt.subplot(1,2,2)
plt.title('ku - bordas verticais')
plt.imshow(Imagem_u, cmap='gray')
plt.show()

#magnitude do gradiente

G = np.sqrt(Imagem_v**2 + Imagem_u**2)

#normalizar G
G_normalizado  = cv2.normalize(G,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.title('Imagem Original')
plt.imshow(imagem_original, cmap='gray')
plt.subplot(1,2,2)
plt.title("Imagem G normalizada - Magnitude do Gradiente")
plt.imshow(G_normalizado, cmap='gray')
plt.show()

plt.figure(figsize=(14,10))
plt.suptitle('Comparação Geral')
plt.subplot(2,2,1)
plt.title('Imagem Original')
plt.imshow(imagem_original, cmap='gray')
plt.subplot(2,2,2)
plt.title('kv - bordas horizontais')
plt.imshow(Imagem_v, cmap='gray')
plt.subplot(2,2,3)
plt.title('ku - bordas verticais')
plt.imshow(Imagem_u, cmap='gray')
plt.subplot(2,2,4)
plt.title("Imagem G normalizada - Magnitude do Gradiente")
plt.imshow(G_normalizado, cmap='gray')
plt.subplots_adjust(hspace=0.7)
plt.show()


plt.hist(G_normalizado.ravel(),bins=256, range=[0,256])
plt.yscale('log')
plt.show()

Tg = 63

imagem_binaria_Em = np.where(G_normalizado>Tg, 255, 0).astype(np.uint8)
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.title('Imagem Binária com limiar que escolhi - Tg = 63')
plt.imshow(imagem_binaria_Em, cmap='gray')
plt.subplot(1,2,2)
plt.title('Imagem magnitude normalizada do Gradiente')
plt.imshow(G_normalizado, cmap='gray')

plt.show()


T_otsu , binaria_otsu = cv2.threshold(G_normalizado,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(14,7))
plt.subplot(2,2,1)
plt.title('Imagem magnitude normalizada do Gradiente')
plt.imshow(G_normalizado, cmap='gray')
plt.subplot(2,2,2)
plt.title('Imagem Binária com limiar que escolhi - Tg = 63')
plt.imshow(imagem_binaria_Em, cmap='gray')
plt.subplot(2,2,3)
plt.title(f"Imagem Binaria com limiar automático de Otsu - T_otsu = {T_otsu}")
plt.imshow(binaria_otsu, cmap='gray')
plt.subplots_adjust(hspace=0.4)
plt.show()

