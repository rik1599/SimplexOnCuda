# Progetto di Programmazione di Architetture parallele
Metodo del simplesso a due fasi in CUDA

Autori: Belliato Riccardo e Tomada Simone

Anno accademico 2021-2022

## Istruzioni di compilazione
Su Windows:
```
.\compile.bat
```
Su Linux/MacOS
```
./compile.sh
```
Lo script crea un eseguibile per c.c. 6.0 e superiori e lo inserisce nella cartella debug 
(So che bisognerebbe usare make, tuttavia sui PC del laboratorio non è installato make e quindi non è possibile verificare se un eventuale file è corretto)

## Relazione in pdf
Si trova [qui](./relazione/relazione.pdf)

## Dataset delle istanze
Le istanze usate per i test si trovano [qui](./data/examples/benchmark_problems)

Per eseguire una istanza tra quelle utilizzate per i benchmark utilizzare i seguenti comandi:
```
cd debug
./SimplexOnCuda -rf ./../data/examples/benchmark_problems/<nome_file>
```

Per lanciare l'intero benchmark:
```
cd debug
./SimplexOnCuda -t
```

Per lanciare un nuovo problema istanziato casualmente
```
cd debug
./SimplexOnCuda -r <numero_variabili> <numero_vincoli> <eventuale_seed>
```
Con l'opzione -rs viene anche salvato un file di testo con questi parametri.
Questo file può essere utilizzato eseguendo SimplexOnCuda con l'opzione -rf e il path del file.

Se invece si vuole fornire un problema specifico creare un file con questo formato
```
3 2           -> Numero di variabili + numero di vincoli
8 10 7        -> Coefficienti della funzione obiettivo
1 3 2 10      -> Coefficienti del primo vincolo + termine noto
1 5 1 8       -> Coefficienti del secondo vincolo + termine noto
...
```

ed eseguire SimplexOnCuda con l'opzione -f e il path del file con il problema.

## Dataset dei tempi di esecuzioni
I risultati dei tempi di esecuzione si trovano [qui](./data/measures/).

