/**
 *	Es5 p1
 *	Calcolo SOMMA di due MATRICI.
 *  1. Memorizzare le matrici in array monodimensionali
 *	2. Configurare il kernel come una griglia bidimensionale di BxB blocchi,
 *	   con blocchi bidimensionali di TxT threads, con TRE diversi valori di T.
 *	   Per ogni configurazione del kernel usata, calcolare il numero di blocchi residenti
 *	   in uno streaming multiprocessor ed il numero di thread attivi, 
 *	   in base alla GPU utilizzata.
 *
 *	Eseguire il programma sviluppato, usando le tre diverse configurazioni del kernel. 
 *	Calcolare i tempi di esecuzione e lo Speed up, al variare della dimensione N del 
 *	problema, con N>1.000 . Utilizzare valori di N multipli di 32, ad esempio 
 *	N=1.024, 2048, 4.096, 8.192, 16.384, ... .
 **/

#include <cuda.h>
#include <stdio.h>
#include <time.h>

//Funzioni d'appoggio
void initializeMatrix(int *array, int n){
	int i;
	for(i=0;i<n;i++)
		 array[i] = rand()%5-2; // range tra -2 e 2
}

void printMatrix(int* matrix, int n){
	int i,j;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			printf("%d\t", matrix[(i * n) + j]);
		}
		printf("\n");
	}
}


void stampaArray(int* array, int n){
	int i;
	for(i=0;i<n;i++)
		printf("%d ", array[i]);
	printf("\n");
}

void equalArray(int* a, int*b, int n){
	int i=0;
  
	while(a[i]==b[i]){
		i++;
  }

	printf("\n1. ");
	if(i<n)
		printf("I risultati dell'host(C_host) e del device(C_device/copy) sono diversi\n");
	else
		printf("I risultati dell'host(C_host) e del device(C_device/copy) coincidono\n");
}

//Seriale
void sommaMatrixCPU(int* A, int* B, int* C, int N){
	for( int i = 0; i < N; i++ ){
		C[i] = A[i] + B[i];
	}
}

//Parallelo
__global__ void sommaMatrixGPU(int* A, int* B, int* C, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int index = j * gridDim.x * blockDim.x + i; // indice/coordinata globale del thread: la relazione tra l'indice dell'array monodimensionale che rappresenta i dati del problema e l'indice del thread è la seguente

	if(index < N)
		C[index] = A[index] + B[index];
}

//risultato tot, Seriale
int sommaTot( int* c, int n ){
	int T = 0;
	for( int i = 0; i < n; i++ )
		T += c[i];
	return T;
}

int main(int argn, char* argv[]){
	int N; 								 // numero totale di elementi dell'array
	int *A_host, *B_host, *C_host; 		 // host data
	int *A_device, *B_device, *C_device; // device data
	int *copy;							 // array in cui copieremo i risultati di C_device
	int nBytes;							 // size in byte di ciascun array
	dim3 gridDim, blockDim;				 // numero di blocchi e numero di thread per blocco

	printf("***\t SOMMA DI DUE MATRICI: \t***\n\tcon griglia di BxB blocchi \n\te blocchi di TxT threads \n");
	printf("Inserisci la dimensione della matrice (multiplo di 32), N:\n");
	scanf("%d",&N);
  	//N = 10000000;
	//printf("Inserisci il numero di Thread per blocco (multiplo di 32), T\n");
	//scanf("%d",&T); //8x8 e 16x16
	blockDim.x = 16;
	blockDim.y = 16;


	//determinazione esatta del numero di blocchi
	gridDim.x = N / blockDim.x + ((N % blockDim.x) == 0 ? 0 : 1);		//gridDim.x in realta e' numBlocks sull'asse x
	gridDim.y = N / blockDim.y + ((N % blockDim.y) == 0 ? 0 : 1);		//gridDim.y in realta e' numBlocks sull'asse y
	printf("gridDim.x: %d, gridDim.y: %d\n", gridDim.x, gridDim.y);
	printf("blockDim.x: %d, blockDim.y: %d\n", blockDim.x, blockDim.y);

	//size in byte di ogni array
	nBytes = N * N * sizeof(int);

	/*allocazione dati sull'host*/
	A_host=(int*)malloc(nBytes);
	B_host=(int*)malloc(nBytes);
	C_host=(int*)malloc(nBytes);
	copy=(int*)malloc(nBytes);

	/*allocazione dati sul device*/
	cudaMalloc((void**)&A_device,nBytes);
	cudaMalloc((void**)&B_device,nBytes);
	cudaMalloc((void**)&C_device,nBytes);

	//inizializzazione dati sull'host
	initializeMatrix(A_host, N * N);
	initializeMatrix(B_host, N * N);

	//copia dei dati dall'host al device
	/*dobbiamo fare in modo che il codice giri sulla GPU*/ 
	cudaMemcpy(A_device, A_host, nBytes, cudaMemcpyHostToDevice); //cudaMemcpy(void *dst, void *src, size_t size, direction): trasferisco i dati dall'host al device(=GPU) 
	cudaMemcpy(B_device, B_host, nBytes, cudaMemcpyHostToDevice);

	//azzeriamo il contenuto della matrice C
	memset(C_host, 0, nBytes);          // sia su host
	cudaMemset(C_device, 0, nBytes);    // sia su device


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	///START SOMMA GPU
	cudaEventRecord(start);     /*salvo il tempo di inizio GPU*/

	//invocazione del kernel
	sommaMatrixGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N * N);
	cudaEventRecord(stop);      /*salvo il tempo di fine GPU*/
	cudaEventSynchronize(stop); /*assicura che tutti i thread siano arrivati all'evento stop prima di registrare il tempo*/
	float elapsed;

	// tempo tra i due eventi in millisecondi
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("tempo GPU=%f\n", elapsed/1000);

	//copia dei risultati dal device all'host
	cudaMemcpy(copy,C_device,nBytes, cudaMemcpyDeviceToHost);


	///START PRODOTTO CPU
	clock_t start_inst = clock();       /*salvo il tempo di inizio CPU*/
	//chiamata alla funzione seriale per il prodotto di due array
	sommaMatrixCPU(A_host, B_host, C_host, N * N);
	///STOP
	clock_t stop_inst = clock();        /*salvo il tempo di fine CPU*/
	double elapsedCPU = (double)(stop_inst - start_inst) / CLOCKS_PER_SEC;

	printf("tempo CPU=%lf\n", elapsedCPU);


	//stampa degli array e dei risultati
	if(N*N<30) {
		printf("\nA_host:\n"); printMatrix(A_host, N );
		printf("B_host:\n"); printMatrix(B_host, N );
		printf("\nC_host:\n"); printMatrix(C_host, N );
		printf("\nC_device(copy):\n"); printMatrix(copy, N );
	}

	//test di correttezza tra 'copy'(sarebbe C_device) e 'C_host'
	equalArray(copy, C_host, N * N);


	//de-allocazione host
	free(A_host);
	free(B_host);
	free(C_host);
	free(copy);

	//de-allocazione device
	cudaFree(A_device);
	cudaFree(B_device);
	cudaFree(C_device);

	exit(0);
}