#pragma once

#include "macro.h"

/** Trova il valore minimo di un vettore con il metodo della full reduction e ritorna la posizione del valore nel vettore 
 * @param g_vet - vettore di valori in memoria globale
 * @param size - dimensione di g_vet
 * @param outIndex - puntatore a variabile su cui verrà scritto l'indice del valore minimo in g_vet
 * 
 * @return il valore minimo del vettore
*/
TYPE minElement(TYPE* g_vet, unsigned int size, unsigned int* outIndex);

/** Stabilisce se un vettore ha tutti i valori >= 0 utilizzando il metodo della full reduction
 * 
 * @param g_vet - vettore di valori in memoria globale
 * @param size - dimensione di g_vet
 * 
 * @return true se tutti i valori di g_vet sono >= 0, false altrimenti
*/
bool isGreaterThanZero(TYPE* g_vet, unsigned int size);

/** Stabilisce se un vettore ha tutti i valori <= 0 utilizzando il metodo della full reduction
 * 
 * @param g_vet - vettore di valori in memoria globale
 * @param size - dimensione di g_vet
 * 
 * @return true se tutti i valori di g_vet sono <= 0, false altrimenti
*/
bool isLessThanZero(TYPE* g_vet, unsigned int size);