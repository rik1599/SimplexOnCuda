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
(So che bisognerebbe usare make, tuttavia utilizzando i PC del laboratorio bisogna arrangiarsi 😊)

## Relazione in pdf
Si trova [qui](./relazione/relazione.pdf)

## Dataset delle istanze
Le istanze usate per i test si trovano [qui](./data/examples/benchmark_problems)

Per eseguire una istanza tra quelle utilizzate per i benchmark utilizzare i seguenti comandi:
```
cd debug
./SimplexOnCuda -rf ./../data/examples/benchmark_problems/<nome_file>
```

## Dataset dei tempi di esecuzioni
I risultati dei tempi di esecuzione si trovano [qui](./data/measures/)

