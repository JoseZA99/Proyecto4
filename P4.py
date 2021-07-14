# > Proyecto 4
# Jose Ignacio Zamora Alvarez (B78505)
# B78505

#Librerias necesarias
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import fft

######## PARTE A ########

# Función que transforma la imagen al modelo de colores RGB

def fuente_info(imagen):
   
    img = Image.open(imagen)
    
    return np.array(img)

# Función que transforma el modelo de colores RGB en una cadena de bits

def rgb_a_bit(imagen):
   
    # Obtener las dimensiones de la imagen
    x, y, z = imagen.shape
    
    # Número total de pixeles
    n_pixeles = x * y * z

    # Convertir la imagen a un vector unidimensional de n_pixeles
    pixeles = np.reshape(imagen, n_pixeles)

 
    bits = [format(pixel,'08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)


# Función que crea la onda modulada en QAM16

def modulador(bits, fc, mpp):
   
    # 1. Parámetros de la 'señal' de información (bits)
    
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora s(t)
    
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)
    portadora_1 = np.cos(2*np.pi*fc*t_periodo)  # Portadora I
    portadora_2 = np.sin(2*np.pi*fc*t_periodo)  # Portadora Q

    # 3. Inicializar la señal modulada s(t)
    
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_I = np.zeros(t_simulacion.shape)    
    senal_Q = np.zeros(t_simulacion.shape) 
    moduladoraI = np.zeros(t_simulacion.shape)  
    moduladoraQ = np.zeros(t_simulacion.shape)   

    # 4. Asignar las formas de onda según los bits de la manera QAM16
    
    j = 0    # Contador de muestreos 
    
    # La idea es recorrer el vector de bits de dos en dos para asignar el valor de A1 y A2
    # Bit b1b2 --> A1 | Bit b3b4 ---> A2 
   
    # Para la señal modulada se asigna la mitad del periodo de muestro a cada bit 
    
    for i in range(0,N,4):
        
        # Portadora I
        #Aqui se analizan las senalas b1b2
        if i < N+4:
            if bits[i] == 0 and bits[i+1] == 0:
                senal_I[j*mpp : (j+1)*mpp] = portadora_1*(-3)
            elif bits[i]== 0 and bits[i+1] == 1:
                senal_I[j*mpp : (j+1)*mpp] = portadora_1*(-1)
            elif bits[i]== 1 and bits[i+1] == 0:
                senal_I[j*mpp : (j+1)*mpp] = portadora_1*(3)
            elif bits[i]==  1 and bits[i+1] == 1:  
                senal_I[j*mpp : (j+1)*mpp] = portadora_1*(1)
        
        # Portadora Q
        #Aqui se analizan las senalas b3b4
            if bits[i+2] == 0 and bits[i+3] == 0:
                senal_Q[j*mpp : (j+1)*mpp] = portadora_2*(3)
            elif bits[i+2]== 0 and bits[i+3] == 1:
                senal_Q[j*mpp : (j+1)*mpp] = portadora_2 *(1)
            elif bits[i+2]== 1 and bits[i+3] == 0:    
                senal_Q[j*mpp : (j+1)*mpp] = portadora_2 *(-3)
            elif bits[i+2]== 1 and bits[i+3] == 1:  
                senal_Q[j*mpp : (j+1)*mpp] = portadora_2 *(-1)
        
        j=j+1
    
    senal_Tx =  senal_I + senal_Q  # Portadora Final S(t) 
    
    
    # 5. Calcular la potencia promedio de la señal modulada
    
    Pm = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    return senal_Tx, Pm, portadora_1,portadora_2 , moduladoraI, moduladoraQ, t_simulacion

# Función que simula un medio no ideal (ruidoso)
def canal_ruidoso(senal_Tx, Pm, SNR):
  
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

# Función que demodula la señal trasmitida por el medio ruidoso 

def demodulador(senal_Rx, portadoraI,portadoraQ, mpp):
 
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits en transmisión
    N =  int(M / mpp) # Se multiplica por dos ya que hay dos bits por muestreo

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demoduladaI = np.zeros(M)
    senal_demoduladaQ = np.zeros(M)
    senal_demodulada = np.zeros(M)
   
   
    
    j =0 # Puntero del array de bits
    
    # Al igual que en la modulación para graficar la señal demodulada cada bit o fragmento se aplica la mitad del periodo 
    for i in range(N):

        
        if j+4 >N:
            break
        # Producto interno de dos funciones
        producto_1 = senal_Rx[(i)*mpp : (i+1)*mpp] * portadoraI                   # Producto de la portadora I
        producto_2 = senal_Rx[(i)*mpp : (i+1)*mpp] * portadoraQ                   # Producto de la portadora Q
        senal_demoduladaI[i*mpp : (i+1)*mpp] = producto_1        # Parte asociada a la portadora I
        senal_demoduladaQ[(i)*mpp : (i+1)*mpp] = producto_2      # Parte asociada a la portadora I

    
        if np.sum(producto_1) >= 0:
            bits_Rx[j] = 1
        if np.sum(producto_2) < 0:
            bits_Rx[j+2] = 1
        if np.max(np.abs(producto_1)) < 2.5:
            bits_Rx[j+1] = 1   
        if np.max(np.abs(producto_2)) < 2.5:
            bits_Rx[j+3] = 1          
        j += 4 
            
    return bits_Rx.astype(int), senal_demoduladaI, senal_demoduladaQ

# Función que reconstruye la imagen 

def bits_a_rgb(bits_Rx, dimensiones):

    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)


#..... Pruebas del código..........

# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 5    # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadoraI, portadoraQ, moduladoraI,moduladoraQ, t_simulacion = modulador(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx,senal_demoduladaI, senal_demoduladaQ = demodulador(senal_Rx, portadoraI, portadoraQ, mpp)



# Se plotean las distintas etapas del proceso

fig, ( ax1,ax2,ax3) = plt.subplots(nrows=3, sharex=True, figsize=(14, 7))


ax1.plot(moduladoraI[0:600], color='r', lw=2) 
ax1.set_ylabel('$b1(t)$')

# La señal modulada por QAM16
ax2.plot(senal_Rx[0:600], color='m', lw=2) 
ax2.set_ylabel('$s(t) + n(t)$')


# La señal modulada al dejar el canal
ax3.plot(senal_Tx[0:600], color='g', lw=2) 
ax3.set_ylabel('$s(t)$')
ax3.set_xlabel('$t$ / milisegundos')





fig, (ax1,ax3,ax2,ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(senal_Tx[0:600], color='g', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por QAM16
ax2.plot(senal_Rx[0:600], color='m', lw=2) 
ax2.set_ylabel('$s(t) + n(t)$')

# La señal demodulada I
ax3.plot(senal_demoduladaI[0:600], color='r', lw=2) 
ax3.set_ylabel('$b1^{\prime}(t)$')

# La señal demodulada Q
ax4.plot(senal_demoduladaQ[0:600], color='b', lw=2) 
ax4.set_ylabel('$b2^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')



# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()


plt.imshow(imagen_Rx)


######### PARTE 2 ###########

# Tiempo de muestreo 
t_muestra = np.linspace(0, 0.1,100)

# Se obtiene el promedio de la señal Tx
PromTx = np.mean(senal_Tx)

# Posibles valores de A
A=[1,-1]

# Formas de onda posibles 

X_t = np.empty((4, len(t_muestra)))	   # 4 funciones del tiempo x(t) 

# Nueva figura 
plt.figure()

# Matriz con los valores de cada función posibles
   
for i in A:
    x1 = i * np.cos(2*(np.pi)*fc*t_muestra) +  i* np.sin(2*(np.pi)*fc*t_muestra)
    x2 = -i * np.cos(2*(np.pi)*fc*t_muestra) +  i* np.sin(2*(np.pi)*fc*t_muestra) 
    X_t[i,:] = x1
    X_t[i+1,:] = x2
    plt.plot(t_muestra,x1, lw=2)
    plt.plot(t_muestra, x2, lw=2)       

#Promedio de las 4 realizaciones en cada instante 
P = [np.mean(X_t[:,i]) for i in range(len(t_muestra))]
plt.plot(t_muestra, P, lw=6,color='k',label='Promedio Realizaciones')

# Graficar el resultado teórico del valor esperado
E = np.mean(senal_Tx)*t_muestra  # Valor esperado de la señal 
plt.plot(t_muestra, E, '-.', lw=3,color='c',label='Valor teórico')

# Mostrar las realizaciones, y su promedio calculado y teórico

plt.title('Realizaciones del proceso aleatorio $X(t)$')
plt.xlabel('$t$')
plt.ylabel('$x_i(t)$')
plt.legend()
plt.show()
er=np.mean(senal_Tx)
t2=np.mean(senal_Tx[0:20])

print('Ergosidad:',er,'y',t2)












