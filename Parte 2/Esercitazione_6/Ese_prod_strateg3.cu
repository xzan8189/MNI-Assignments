#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>

//Funzioni d'appoggio
void initializeArray(float *array, int n){
	int i;
  for (i = 0; i < n; i++) {
    array[i] = ((float)rand() * 4 / (float)RAND_MAX) - 2; //reali nell'intervallo (-2,+2)
  }
}

void printArray_float(float *array, int n){
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
__global__ void prodottoScalareGPU3(float *a, float *b, float *c, int n) {
  /* vettore shared che conterrà i prodotti effettuati dai thread di un blocco */
  extern __shared__ float v[];

  /* ogni thread ricava gli indici su cui deve lavorare */
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int id = threadIdx.x;

  if (index < n) {
    /* prodotto componenti dei due vettori */
    v[id] = a[index] * b[index];

    /* sincronizzazione dei thread */
    __syncthreads();

    /* Somma parallela dei prodotti effettuati dai thread di un blocco */
    for (int dist = blockDim.x; dist > 1;) {
      dist = dist / 2;
      if (id < dist) {
        v[id] = v[id] + v[id + dist];
      }
      __syncthreads();
    }

    if (id == 0) {
      /* il thread 0 ha la somma finale dei prodotti effettuati dai thread di un blocco */
      c[blockIdx.x] = v[0];
    }
  }
}

void prodottoScalareSequenziale(float *a, float *b, float *prodotto_scalare_host, int n) {
  for (int i = 0; i < n; i++) {
    *prodotto_scalare_host += a[i] * b[i];
  }
}

int main(void) {
  int n;
  float *A_host, *B_host, *copy;                // host data
  float *A_device, *B_device, *C_device;        // device data
  float prod_scalare_device = 0.0, prodotto_scalare_host = 0.0;
  double nBytes;          // size in byte di ciascun array
	dim3 gridDim, blockDim; // numero di blocchi e numero di thread per blocco

	printf("***\t PRODOTTO SCALARE DI DUE VETTORI - 3 STRATEGIA \t***\n");
  printf("Inserisci numero elementi dei vettori: ");
  scanf("%d", &n);  //1000000, 2000000, 4000000, 8000000, 16000000
  blockDim.x = 64; //32, 64, 128

	//determinazione esatta del numero di blocchi
  gridDim.x = n/blockDim.x + ((n%blockDim.x)==0?0:1);   //gridDim in realta e' numBlocks
  printf("blockDim: %d\n", blockDim.x);
  printf("gridDim: %d\n\n", gridDim.x);

  //size in byte di ogni array
  nBytes = n * sizeof(float);

  /*allocazione dati sull'host*/
  A_host = (float*) malloc(nBytes);
  B_host = (float*) malloc(nBytes);
  copy = (float*) malloc(gridDim.x * sizeof(float));

  /* Allocazine memoria device */
  cudaMalloc((void **) &A_device, nBytes);
  cudaMalloc((void **) &B_device, nBytes);
  cudaMalloc((void **) &C_device, gridDim.x * sizeof(float));

  //inizializzazione dati sull'host
  //srand(time(NULL));
  initializeArray(A_host, n);   //reali nell'intervallo (-2,+2)
  initializeArray(B_host, n);

  /* Copia vettori da host a device */
  cudaMemcpy(A_device, A_host, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(B_device, B_host, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(C_device, copy, gridDim.x * sizeof(float), cudaMemcpyHostToDevice); //VEDERE SE SI PUò TOGLIERE

	///START PRODOTTO SCALARE CPU
  clock_t inizio = clock();   /*salvo il tempo di inizio CPU*/
  prodottoScalareSequenziale(A_host, B_host, &prodotto_scalare_host, n);
  clock_t fine = clock();     /*salvo il tempo di fine CPU*/
  float elapsedCPU = (float)(fine - inizio) / CLOCKS_PER_SEC;    

	printf("tempo CPU=%lf\n", elapsedCPU);
  ///STOP CPU

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	///START PRODOTTO SCALARE GPU
  cudaEventRecord(start);   /*salvo il tempo di inizio GPU*/

  /* invocazione dinamica del kernel */
  prodottoScalareGPU3<<<gridDim, blockDim, blockDim.x>>>(A_device, B_device, C_device, n);

	//copia dei risultati dal device all'host
  cudaMemcpy(copy, C_device, gridDim.x * sizeof(float), cudaMemcpyDeviceToHost);

  /* somma sull'host */
  for (int i = 0; i < gridDim.x; i++) {
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
  if (n <= 20) {
		printf("\nA_host: "); printArray_float(B_host, n);
		printf("B_host: "); printArray_float(A_host, n);
		printf("C_device(copy): "); printArray_float(copy, gridDim.x);
  }
  
  printf("\nprodotto_scalare_host (host): %f\n", prodotto_scalare_host);
  printf("prod_scalare_device (device): %f\n", prod_scalare_device);

  /* free della memoria */
  free(A_host);
  free(B_host);
  free(copy);
  cudaFree(A_device);
  cudaFree(B_device);
  cudaFree(C_device);

  return 0;
}
