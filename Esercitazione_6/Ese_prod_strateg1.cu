#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

//Funzioni d'appoggio
void initializeArray(double *array, int n){
	int i;
  for (i = 0; i < n; i++) {
    array[i] = ((double)rand() * 4 / (double)RAND_MAX) - 2; //reali nell'intervallo (-2,+2)
  }
}

void printArray_double(double *array, int n){
	int i;
	for(i=0;i<n;i++)
		printf("%f ", array[i]);
	printf("\n");
}

void equalValues(double a, double b){
	printf("\n1. ");
	if(a != b)
		printf("\033[0;31mI risultati dell'host e del device sono diversi\033[0;30m\n");
	else
		printf("\033[0;32mI risultati dell'host e del device coincidono\033[0;30m\n");
}

//Funzioni core
__global__ void prodottoGPU(double *u, double *v, double *w, int n) {
    /* Ogni thread ricava su quali elementi deve lavorare */
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        w[index] = v[index] * u[index];
    }
}

void prodottoScalareSequenziale(double *u, double *v, double *prodotto_scalare_host, int n) {
    for (int i = 0; i < n; i++) {
        *prodotto_scalare_host += u[i] * v[i];
    }
}

int main(void) {
  int N;    //dimensione vettori
  double *A_host, *B_host, *copy;                   // host data
  double *A_device, *B_device, *C_device;           // device data
  double prod_scalare_device = 0.0, prodotto_scalare_host = 0.0;
  double nBytes;          // size in byte di ciascun array
	dim3 gridDim, blockDim; // numero di blocchi e numero di thread per blocco


	printf("***\t PRODOTTO SCALARE DI DUE VETTORI - 1 STRATEGIA \t***\n");
  printf("Inserisci numero elementi dei vettori: ");
  scanf("%d", &N);  //1000000, 2000000, 4000000, 8000000, 16000000
  blockDim.x = 64; //32, 64, 128
  
	//determinazione esatta del numero di blocchi
  gridDim.x = N/blockDim.x + ((N%blockDim.x)==0?0:1);   //gridDim in realta e' numBlocks
  printf("blockDim: %d\n", blockDim.x);
  printf("gridDim: %d\n\n", gridDim.x);

  //size in byte di ogni array
  nBytes = N * sizeof(double);

  /*allocazione dati sull'host*/
  A_host = (double*)malloc(nBytes);
  B_host = (double*)malloc(nBytes);
  copy = (double*)malloc(nBytes);

  /*allocazione dati sul device*/
  cudaMalloc((void **) &A_device, nBytes);
  cudaMalloc((void **) &B_device, nBytes);
  cudaMalloc((void **) &C_device, nBytes);

  //inizializzazione dati sull'host
  //srand(time(NULL));
  initializeArray(A_host, N);   //reali nell'intervallo (-2,+2)
  initializeArray(B_host, N);

	//copia dei dati dall'host al device
  cudaMemcpy(A_device, A_host, nBytes, cudaMemcpyHostToDevice);   //cudaMemcpy(void *dst, void *src, size_t size, direction): trasferisco i dati dall'host al device(=GPU) 
  cudaMemcpy(B_device, B_host, nBytes, cudaMemcpyHostToDevice);

	///START PRODOTTO SCALARE CPU
  clock_t inizio = clock();   /*salvo il tempo di inizio CPU*/
  prodottoScalareSequenziale(A_host, B_host, &prodotto_scalare_host, N);
  clock_t fine = clock();     /*salvo il tempo di fine CPU*/
  float elapsedCPU = (float)(fine - inizio) / CLOCKS_PER_SEC;    

	printf("tempo CPU=%lf\n", elapsedCPU);
  ///STOP CPU

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	///START PRODOTTO SCALARE GPU
  cudaEventRecord(start);   /*salvo il tempo di inizio GPU*/

	//invocazione del kernel
  prodottoGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N);

	//copia dei risultati dal device all'host
  cudaMemcpy(copy, C_device, nBytes, cudaMemcpyDeviceToHost);

  //somma seriale su CPU (IL DEVICE FA LA SOMMA SULL'HOST, MA NON POTREBBE ESSERE FATTA SUL DEVICE?)
  for (int i = 0; i < N; i++) {
      prod_scalare_device += copy[i];
  }
  ///STOP GPU
      
  cudaEventRecord(stop);    // tempo di fine
  cudaEventSynchronize(stop);
  float elapsed;

	// tempo tra i due eventi in millisecondi
  cudaEventElapsedTime(&elapsed, start, stop);    //in ms
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

	printf("tempo GPU=%f\n", elapsed/1000);   //in secondi

	//stampa degli array e dei risultati
	if(N<20) {
		printf("\nA_host: "); printArray_double(B_host, N);
		printf("B_host: "); printArray_double(A_host, N);
		printf("C_device(copy): "); printArray_double(copy, N);
	}

  printf("\nprodotto_scalare_host (host): %f\n", prodotto_scalare_host);
  printf("prod_scalare_device (device): %f\n", prod_scalare_device);

	//test di correttezza tra 'prod_scalare_device' e 'prodotto_scalare_host'
	equalValues(prodotto_scalare_host, prod_scalare_device);

  /* free della memoria host e device */
  free(A_host);
  free(B_host);
  free(copy);
  cudaFree(A_device);
  cudaFree(B_device);
  cudaFree(C_device);

  return 0;
}

