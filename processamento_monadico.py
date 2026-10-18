import cv2
import numpy as np
import matplotlib.pyplot as plt

imagem_uint8 = cv2.imread('img_processar_monadico.jpg', cv2.IMREAD_GRAYSCALE)
imagem_double = imagem_uint8.astype(np.float64)/255.0 #dividir por 255 pra deixar no intervalo zero até 1
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("imagem uint8")
plt.imshow(imagem_uint8, cmap='gray')
plt.subplot(1,2,2)
plt.title('imagem double')
plt.imshow(imagem_double, cmap='gray')
plt.show()

print(imagem_uint8.dtype)
print(imagem_double.dtype)

maxima_uint8 = np.max(imagem_uint8)
minima_uint8 = np.min(imagem_uint8)
media_uint8 = np.mean(imagem_uint8)

maxima_double = np.max(imagem_double)
minima_double = np.min(imagem_double)
media_double = np.mean(imagem_double)

print(f'Imagem uint8: min: {minima_uint8}, max: {maxima_uint8}, media: {media_uint8}')
print(f'Imagem double: min: {minima_double}, max: {maxima_double}, media: {media_double}')

beta_brilho_uint8 = 60
beta_brilho_double = beta_brilho_uint8/255

#aplicar brilho
brilho_uint8 = np.clip(imagem_uint8.astype(np.int16)+beta_brilho_uint8, 0, 255).astype(np.uint8)
brilho_double = np.clip(imagem_double+beta_brilho_double, 0.0, 1.0)

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.title("imagem uint8 com brilho")
plt.imshow(brilho_uint8, cmap='gray')
plt.subplot(1,3,2)
plt.title('imagem double com brilho')
plt.imshow(brilho_double, cmap='gray')
plt.subplot(1,3,3)
plt.title('imagem original')
plt.imshow(imagem_uint8, cmap='gray')
plt.show()

alpha = 1.5
contraste_uint8 = cv2.convertScaleAbs(imagem_uint8, alpha=alpha, beta=beta_brilho_uint8)
contraste_double = np.clip(alpha*imagem_double + beta_brilho_double, 0.0, 1.0)

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.title(f"imagem uint8 - contraste  beta {beta_brilho_uint8} e alpha {alpha}")
plt.imshow(contraste_uint8, cmap='gray')
plt.subplot(1,2,2)
plt.title(f'imagem double - contraste beta {beta_brilho_double} e alpha {alpha}')
plt.imshow(contraste_double, cmap='gray')
plt.subplots_adjust(hspace=0.5)
plt.show()

negativo_uint8 = 255 - imagem_uint8

negativo_double = 1 - imagem_double

plt.figure(figsize=(14,5))
plt.subplot(1,3,1)
plt.title('imagem original')
plt.imshow(imagem_uint8, cmap='gray')
plt.subplot(1,3,2)
plt.title(f"imagem uint8 - negativo")
plt.imshow(negativo_uint8, cmap='gray')
plt.subplot(1,3,3)
plt.title(f'imagem double - negativo')
plt.imshow(negativo_double, cmap='gray')
plt.subplots_adjust(hspace=0.5)
plt.show()

#posterização

L_4 = 4
delta_4 = 256//L_4

posterizado_4 = (imagem_uint8//delta_4)*delta_4
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.title('imagem original')
plt.imshow(imagem_uint8, cmap='gray')
plt.subplot(1,2,2)
plt.title('imagem posterizada L = 4 níveis de intensidade')
plt.imshow(posterizado_4, cmap='gray')
plt.show()

L_8 = 8
delta_8 = 256//L_8
posterizado_8 = (imagem_uint8//delta_8)*delta_8


L_16 = 16
delta_16 = 256//L_16

posterizado_16 = (imagem_uint8//delta_16)*delta_16
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title('imagem posterizada L = 4 níveis de intensidade')
plt.imshow(posterizado_4, cmap='gray')
plt.subplot(1,3,2)
plt.title('imagem posterizada L = 8 níveis de intensidade')
plt.imshow(posterizado_8, cmap='gray')
plt.subplot(1,3,3)
plt.title('imagem posterizada L = 16 níveis de intensidade')
plt.imshow(posterizado_16, cmap='gray')
plt.subplots_adjust(hspace=1.4)
plt.show()

plt.figure(figsize=(22,5))
plt.subplot(1,5,1)
plt.title('imagem original uint8')
plt.imshow(imagem_uint8, cmap='gray')
plt.subplot(1,5,2)
plt.title('Imagem uint8 - brilho')
plt.imshow(brilho_uint8, cmap='gray')
plt.subplot(1,5,3)
plt.title('Imagem uint8 - contraste')
plt.imshow(contraste_uint8, cmap='gray')
plt.subplot(1,5,4)
plt.title("imagem uint8 - negativo")
plt.imshow(negativo_uint8,cmap='gray')
plt.subplot(1,5,5)
plt.title("posterizada 4 níveis")
plt.imshow(posterizado_4, cmap='gray')
plt.subplots_adjust(hspace=1.7)
plt.show()

r, c = 0, 0

p_uint8_antes = imagem_uint8[r, c]
p_double_antes = imagem_double[r, c]

p_uint8_depois = np.clip(p_uint8_antes.astype(np.int16)+beta_brilho_uint8, 0, 255).astype(np.uint8)
p_double_depois = np.clip(p_double_antes+beta_brilho_double, 0.0, 1.0)


print("[Representação uint8 (0 a 255)]")
print(f" - Antes da operação : {p_uint8_antes}")
print( f" - Cálculo : {p_uint8_antes} + {beta_brilho_uint8} = {int(p_uint8_antes) + beta_brilho_uint8}" )
print(f" - Depois da operação : {p_uint8_depois}")
print("[Representação double (0.0 a 1.0)]")
print(f" - Antes da operação : {p_double_antes:.4f}")
print( f" - Cálculo : {p_double_antes:.4f} + {beta_brilho_double:.4f} = {p_double_antes + beta_brilho_double:.4f}" )
print(f" - Depois da operação : {p_double_depois:.4f}")

# Em uint8, somas que ultrapassam 255 saturam no limite máximo (255), perdendo a variação de brilho acima do teto em double, os valores são mantidos em ponto flutuante contínuo e fica em 1.0
# e o mesmo se negativo em uint8 vai até zero, mesmo ficando negativo e double tambem
#quando esses cenarios ocorrem dizemos que houve saturação - numero fora da faixa permitida