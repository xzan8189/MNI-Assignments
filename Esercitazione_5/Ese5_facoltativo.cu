/**
 *	Es5 p2
 *	Calcolo SOMMA di due MATRICI considerando matrici rettangolari N x M 
 *	con configurazione del kernel data da una griglia bidimensionale 
 *	di (Bx,By) blocchi, con blocchi bidimensionali di (Tx,Ty) threads 
 *  
 **/

/*

    !Compilazione: nvcc Ese5_Facoltativo.cu -o Ese5_Facoltativo 
    !Esecuzione:    ./Ese5_Facoltativo
*/




#include <cuda.h>
#include <stdio.h>
#include <time.h>

//Funzioni d'appoggio
void initializeMatrix(int *array, int n){
	int i;
	for(i=0;i<n;i++)
		 array[i] = rand()%5-2; // range tra -2 e 2
}

void printMatrix(int* matrix, int n, int m){
	int i,j;
	for(i = 0; i < n; i++){
		for(j = 0; j < m; j++){
			printf("%d\t", matrix[(i * m) + j]);
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
void sommaMatrixCPU(int* A, int* B, int* C, int N, int M){
	for( int i = 0; i < N; i++ ){
		for(int j=0; j < M; j++){
			C[i * M + j] = A[i * M + j] + B[i * M + j];
		}
		
	}
}

//Parallelo
__global__ void sommaMatrixGPU(int* A, int* B, int* C, int dim){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// o dici M o dici gridDim.y * blockDim.y è la stessa cosa
	int index = j * gridDim.x * blockDim.x + i; // indice/coordinata globale del thread: la relazione tra l'indice dell'array monodimensionale che rappresenta i dati del problema e l'indice del thread è la seguente

	if(index < dim)
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
	int N, M; 							 // numero totale di elementi dell'array
	int *A_host, *B_host, *C_host; 		 // host data
	int *A_device, *B_device, *C_device; // device data
	int *copy;							 // array in cui copieremo i risultati di C_device
	int nBytes;							 // size in byte di ciascun array
	dim3 gridDim, blockDim;				 // numero di blocchi e numero di thread per blocco

	printf("***\t SOMMA DI DUE MATRICI: \t***\n\tcon griglia di B1xB2 blocchi \n\te blocchi di T1xT2 threads \n");
	printf("Inserisci il numero di righe (multiplo di 32), N:\n");
	scanf("%d",&N);
    printf("Inserisci il numero di colonne (multiplo di 32), M:\n");
    scanf("%d",&M);
  	

    //Configurazione ottimale del Kernel
	blockDim.x = 8;
	blockDim.y = 8;


	//determinazione esatta del numero di blocchi
	gridDim.x = N / blockDim.x + ((N % blockDim.x) == 0 ? 0 : 1);		//gridDim.x in realta e' numBlocks sull'asse x
	gridDim.y = M / blockDim.y + ((M % blockDim.y) == 0 ? 0 : 1);		//gridDim.y in realta e' numBlocks sull'asse y
	printf("gridDim.x: %d, gridDim.y: %d\n", gridDim.x, gridDim.y);
	printf("blockDim.x: %d, blockDim.y: %d\n", blockDim.x, blockDim.y);

	//size in byte di ogni array
	nBytes = N * M * sizeof(int);

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
	initializeMatrix(A_host, N * M);
	initializeMatrix(B_host, N * M);

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
	sommaMatrixGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, M*N);
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
	sommaMatrixCPU(A_host, B_host, C_host, N, M);
	///STOP
	clock_t stop_inst = clock();        /*salvo il tempo di fine CPU*/
	double elapsedCPU = (double)(stop_inst - start_inst) / CLOCKS_PER_SEC;

	printf("tempo CPU=%lf\n", elapsedCPU);


	//stampa degli array e dei risultati
	if(N*M<30) {
    printf("\nA_host:\n"); printMatrix(A_host, N, M);
		printf("B_host:\n"); printMatrix(B_host, N, M );
		printf("\nC_host:\n"); printMatrix(C_host, N, M );
		printf("\nC_device(copy):\n"); printMatrix(copy, N, M );
	}

	//test di correttezza tra 'copy'(sarebbe C_device) e 'C_host'
	equalArray(copy, C_host, N * M);


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