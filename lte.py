import numpy as np

##############################################################################
#                              SUBRUTINAS
##############################################################################
#definimos una funcion para calcular la función de partición a una T 
'''pasamos el array de temperaturas a cada capa, la degeneración de los niveles
    y la energia de estos en eV. calculamos la función de partición del ión 
    correspondiente a cada temperatura del array y devolvemos un array de U'''
def U(T_c,gi,ni):
    k = 8.6173324e-5 #eV/K
    #calculamos la U 
    u = np.sum(gi*np.exp(-ni/(k*T_c)))
    return u

#definimos la rutina para calcular la densidad electronica con la presion
def dens_e(Pe,T):
    k = 1.3806488e-16 #erg/K
    n = Pe/(k*T)
    return n

#calculo de la eq de saha para obtener cada relación entre iones de un átomo
def saha(ne,T,U1,U2,X1_2):
    k = 8.6173324e-5 #eV/K
    return 2.07e-16*ne*(U1/U2)*T**(-3/2)*np.exp(X1_2/(k*T))

#calculo de las densidades de cada estado de excitacion de un ión
def boltzman(Nion,T,g,ui,Xl_u):
    k = 8.6173324e-5 #eV/K
    return (g/ui)*np.exp(-Xl_u/(k*T))*Nion