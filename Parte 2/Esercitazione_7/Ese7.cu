/*

    !Compilazione:  nvcc Ese7.cu -o Ese7 -lcublas  
    !Esecuzione:    ./Ese7
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


//Funzioni d'appoggio
void initializeArray(float *array, int m){
	int i;
	for(i=0; i<m; i++)
		array[i] = ((float)rand() * 4 / (float)RAND_MAX) - 2;
}

void prodottoCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        *c += a[i] * b[i];
    }
}

void stampaArray(float* array, int n){
	int i;
	for(i=0;i<n;i++)
		printf("%f ", array[i]);
	printf("\n");
}

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int M;
    float* h_a = 0;     // Host array a
    float* d_a;         // Device array a
    float* h_b = 0;     // Host array b
    float *d_b;         // Device array b
    float resultGPU = 0;   //Risultato finale GPU
    float resultCPU = 0; // Risultato finale CPU

    
    printf("Inserisci la dimensione M dei vettori:");
    scanf("%d", &M); //1000000, 2000000, 4000000, 8000000, 16000000

    h_a = (float *)malloc (M * sizeof (*h_a));      // Alloco h_a e lo inizializzo
    if (!h_a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    initializeArray(h_a, M);
    
    h_b = (float *)malloc (M * sizeof (*h_b));  // Alloco h_b e lo inizializzo
    if (!h_b) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    initializeArray(h_b, M);

    ///START PRODOTTO CPU
    clock_t start_inst = clock();       /*salvo il tempo di inizio CPU*/ 
    //chiamata alla funzione seriale per il prodotto scalare tra i due vettori
    prodottoCPU(h_a , h_b, &resultCPU, M);
    ///STOP
	clock_t stop_inst = clock();        /*salvo il tempo di fine CPU*/
    double elapsedCPU = (double)(stop_inst - start_inst) / CLOCKS_PER_SEC;  

    printf("tempo CPU=%lf\n", elapsedCPU);

    cudaStat = cudaMalloc ((void**)&d_a, M*sizeof(*h_a));       // Alloco d_a
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    cudaStat = cudaMalloc ((void**)&d_b, M*sizeof(*h_b));       // Alloco d_b
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    // creo l'handle prima di utilizzare qualsiasi operazione CUBLAS in modo da creare un handle specifico della libreria per
    // la gestione delle informazioni e del relativo contesto in cui essa opera.
    stat = cublasCreate(&handle);               // Creo l'handle per cublas
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }


    // la funzione cublasSetVector ci consente di copiare il vettore h_a che abbiamo instanziato sulla CPU
    // sul vettore d_a sulla GPU ( equivalente a MemCopy in Cuda)
    stat = cublasSetVector(M,sizeof(float),h_a,1,d_a,1);    // Setto h_a su d_a
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_a);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    // la funzione cublasSetVector ci consente di copiare il vettore h_b che abbiamo instanziato sulla CPU
    // sul vettore d_b sulla GPU ( equivalente a MemCopy in Cuda)
    stat = cublasSetVector(M,sizeof(float),h_b,1,d_b,1);    // Setto h_b su d_b
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEvent_t start, stop; // eventi per il calcolo del tempo di esecuzione
    float elapsed = 0.0;
    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ///START PRODOTTO GPU
    cudaEventRecord(start); /*salvo il tempo di inizio GPU*/

    stat = cublasSdot(handle,M,d_a,1,d_b,1,&resultGPU);        // Calcolo il prodotto tramite una libreria di Cublas e la cosa positiva (ma anche negativa) è che non si setta 'gridDim' e 'blockDim'
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed cublasSdot");
        cudaFree (d_a);
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop); /*salvo il tempo di fine GPU*/
    cudaEventSynchronize(stop); /*assicura che tutti i thread siano arrivati all'evento stop  prima di registrare il tempo*/
    // tempo tra i due eventi in millisecondi
    cudaEventElapsedTime(&elapsed, start, stop);
    
    printf("tempo GPU=%lf\n", elapsed/1000);

     //stampa degli array e dei risultati
	if(M<20) {
		printf("\nh_a: "); stampaArray(h_a,M);
		printf("\nh_b: "); stampaArray(h_b,M);
	}
   
    //test di correttezza tra 'resultCPU' e 'resultGPU'
		printf("\nresultCPU: %f", resultCPU); 
		printf("\nresultGPU: %f\n" , resultGPU);

    /* Rilascio degli eventi */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree (d_a);     // Dealloco d_a
    cudaFree (d_b);     // Dealloco d_b
    
    cublasDestroy(handle);  // Distruggo l'handle
    
    free(h_a);      // Dealloco h_a
    free(h_b);      // Dealloco h_b    
    return EXIT_SUCCESS;
}

