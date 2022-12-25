/**
 *	Es4 p1
 *	Calcolo prodotto scalare di due vettori
 *  1. Configurare il kernel come una griglia monodimensionale di D
 *	- Il device nel kernel calcola il vettore w prodottoCompPerComp
 *	- l'host calcola la somma dei singoli elementi
 *	Usare blocchi monodimensionali di 32 threads
 *	Contemplare sia il caso N < che N > 32D (=numero totale di threads)
 *
 **/

#include <cuda.h>
#include <stdio.h>
#include <time.h>

//Funzioni d'appoggio
void initializeArray(int *array, int n){
	int i;
	for(i=0;i<n;i++)
		array[i] = rand()%5-2;
}

void stampaArray(int* array, int n){
	int i;
	for(i=0;i<n;i++)
		printf("%d ", array[i]);
	printf("\n");
}

void equalArray(int* a, int*b, int n){
	int i=0;
	while(a[i]==b[i])
		i++;

	printf("\n1. ");
	if(i<n)
		printf("\033[0;31mI risultati dell'host(C_host) e del device(C_device/copy) sono diversi\033[0;30m\n");
	else
		printf("\033[0;32mI risultati dell'host(C_host) e del device(C_device/copy) coincidono\033[0;30m\n");
}

//Seriale
void prodottoCPU(int *a, int *b, int *c, int n){
	int i;
	for(i=0;i<n;i++)
		c[i]=a[i]*b[i];
}

//Parallelo
__global__ void prodottoGPU(int* a, int* b, int* c, int n){
	int index = threadIdx.x + blockIdx.x * blockDim.x;    // indice/coordinata globale del thread: la relazione tra l'indice dell'array monodimensionale che rappresenta i dati del problema e l'indice del thread è la seguente
	if(index < n)
		c[index] = a[index]*b[index];
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
	dim3 gridDim, blockDim; // numero di blocchi e numero di thread per blocco

	printf("***\t PRODOTTO SCALARE DI DUE ARRAY \t***\n");
	/*printf("Inserisci il numero di elementi dei vettori\n");
	scanf("%d",&N); */
  N = 10;
  printf("N: %d\n", N);
	//printf("Inserisci il numero di Thread per blocco\n");
	//scanf("%d",&blockDim); 
	blockDim.x=64; //32, 64, 128

	//determinazione esatta del numero di blocchi
	gridDim = N/blockDim.x + ((N%blockDim.x)==0?0:1);		//gridDim in realta e' numBlocks

	//size in byte di ogni array
	nBytes=N*sizeof(int);

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
  //srand(time(NULL));
	initializeArray(A_host, N);
	initializeArray(B_host, N);

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

	///START PRODOTTO GPU
	cudaEventRecord(start);     /*salvo il tempo di inizio GPU*/
    
	//invocazione del kernel
	prodottoGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N);
	cudaEventRecord(stop);      /*salvo il tempo di fine GPU*/
  cudaEventSynchronize(stop); /*assicura che tutti i thread siano arrivati all'evento stop  prima di registrare il tempo*/
	float elapsed;

	// tempo tra i due eventi in millisecondi
	cudaEventElapsedTime(&elapsed, start, stop);    //ms
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("tempo GPU=%f (s)\n", elapsed/1000);   //in secondi

	//copia dei risultati dal device all'host
	cudaMemcpy(copy,C_device,nBytes, cudaMemcpyDeviceToHost);


	///START PRODOTTO CPU
	clock_t start_inst = clock();       /*salvo il tempo di inizio CPU*/
	//chiamata alla funzione seriale per il prodotto di due array
	prodottoCPU(A_host, B_host, C_host, N);
	///STOP
	clock_t stop_inst = clock();        /*salvo il tempo di fine CPU*/
	double elapsedCPU = (double)(stop_inst - start_inst) / CLOCKS_PER_SEC;

	printf("tempo CPU=%lf (s)\n", elapsedCPU);


	//stampa degli array e dei risultati
	if(N<20) {
		printf("\nA_host: "); stampaArray(A_host,N);
		printf("B_host: "); stampaArray(B_host,N);
		printf("\nC_host: \t"); stampaArray(C_host,N);
		printf("C_device(copy): "); stampaArray(copy,N);
	}

	//test di correttezza tra 'copy'(sarebbe C_device) e 'C_host'
	equalArray(copy, C_host, N);

    //Somma totale sull'host
	printf("2. Eseguo la somma totale sull'host...\n");
	int result_GPU = sommaTot(copy, N);
  int result_CPU = sommaTot(C_host, N);
	printf("   La somma totale GPU è %d\n", result_GPU);
	printf("   La somma totale CPU è %d\n", result_CPU);


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