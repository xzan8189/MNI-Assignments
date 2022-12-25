/*

    !Compilazione:  nvcc ese7_facoltativo.cu -o ese7_facoltativo -lcublas  
    !Esecuzione:    ./ese7_facoltativo
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

//funzioni ausiliarie
void equalArray(float* a, float* b, int n){
	int i=0;
  
	while(a[i]==b[i]){
		i++;
  }

	printf("\n1. ");
	if(i<n)
		printf("\033[0;31mI risultati dell'host(C_host) e del device(C_device/copy) sono diversi\033[0;30m\n");
	else
		printf("\033[0;32mI risultati dell'host(C_host) e del device(C_device/copy) coincidono\033[0;30m\n");
}

//stampa matrice
void print_matrix_float(float matrix[], int ROW_SIZE, int COL_SIZE) {
    for(int i = 0; i < ROW_SIZE; i++) {
        for(int j = 0; j < COL_SIZE; j++)
            printf("%.2lf\t", matrix[j+(i*COL_SIZE)]);
        printf("\n");
    }
}

//stampa array
void print_array_float(float arr[], int dim) {
    for (int i=0; i<dim; i++)
        printf("%.2f ", arr[i]);
    printf("\n");
}

/*Genera matrice*/
void generate_matrix(float* A, int ROW_SIZE, int COL_SIZE) {
    int m=ROW_SIZE, n=COL_SIZE;
    for (int i=0;i<m;i++) {
        for(int j=0;j<n;j++) {
            if (j==0) {
                A[i*n+j]= 1.0/(i+1)-1;
            } else {
                A[i*n+j]= 1.0/(i+1)-pow(1.0/2.0,j); 
            }
        }
    }
}

//prodotto matrice-vettore sequenziale
void prod_mat_vett(float w[], float *a, int ROWS, int COLS, float v[])
{
    int i, j;
    for(i=0;i<ROWS;i++)
    {
        w[i]=0;
        for(j=0;j<COLS;j++)
        { 
            w[i] += a[i*COLS+j]* v[j];
        } 
    }    
}



int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int N;                   //dimensione della matrice e del vettore
    float* h_matrix = 0;     // Host matrix 
    float* d_matrix;         // Device matrix 
    float* h_vector = 0;     // Host vector 
    float* d_vector;         // Device vector
    float* h_result = 0;     // Risultato GPU Host 
    float* d_result = 0;     // Risultato GPU Device
    float* resultCPU = 0;    // Risultato CPU
	
    printf("Inserisci la dimensione della matrice e la dimensione del vettore: ");
    scanf("%d", &N); //1024, 2048, 4096, 8192, 16384

    h_matrix = (float *)malloc (N * N * sizeof (*h_matrix));  // Alloco h_matrix
    if (!h_matrix) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    
    h_vector = (float *)malloc (N * sizeof (*h_vector));  // Alloco h_vector
    if (!h_vector) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    h_result = (float *)malloc (N * sizeof(*h_result));  // Alloco result
    if (!h_result) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    // Alloco il risultato della CPU
    resultCPU = (float *)malloc ( N * sizeof(*resultCPU));   
    if (!resultCPU) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }


    /*Genero matrice iniziale*/
     generate_matrix(h_matrix, N, N);
        if (N<=10) {
            printf("Matrice h_matrix = \n"); 
            print_matrix_float(h_matrix, N, N);
            printf("\n\n");
        }

    //Genero vettore
        for (int j=0;j<N;j++) {
            h_vector[j]=j;  
        }

        if (N<=10) {
            printf("Vettore h_vector = \n");     
            print_array_float(h_vector, N);
            printf("\n\n");
        }

    ///START PRODOTTO CPU
    clock_t start_inst = clock();       /*salvo il tempo di inizio CPU*/ 
    //chiamata alla funzione seriale per il prodotto scalare tra i due vettori
    prod_mat_vett(resultCPU, h_matrix, N, N, h_vector);
    ///STOP
	clock_t stop_inst = clock();        /*salvo il tempo di fine CPU*/
    double elapsedCPU = (double)(stop_inst - start_inst) / CLOCKS_PER_SEC;  

    printf("tempo CPU=%lf\n", elapsedCPU);

    // Alloco d_matrix
    cudaStat = cudaMalloc ((void**)&d_matrix, N * N * sizeof(*h_matrix));   
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }

    // Alloco d_vector
    cudaStat = cudaMalloc ((void**)&d_vector, N * sizeof(*h_vector));  
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc ((void**)&d_result, N * sizeof(*h_result));  // Alloco d_result
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    stat = cublasCreate(&handle);   // Creo l'handle per cublas
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    stat = cublasSetMatrix(N, N, sizeof(float), h_matrix, N, d_matrix, N); // Setto h_matrix su d_matrix
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed matrix");
        cudaFree (d_matrix);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    stat = cublasSetVector(N, sizeof(float), h_vector, 1, d_vector, 1);    // Setto h_vector su d_vector
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed vector");
        cudaFree (d_vector);
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

    //scalari utlizzati per la moltiplicazione
    float alpha = 1.0;
    float beta = 1.0;
    stat = cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, d_matrix, N, d_vector, 1, &beta, d_result, 1); // calcolo prodotto matrice-vettore
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed cublasSdot");
        cudaFree (d_matrix);
        cudaFree (d_vector);
        cudaFree (d_result);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop); /*salvo il tempo di fine GPU*/
    cudaEventSynchronize(stop); /*assicura che tutti i thread siano arrivati all'evento stop  prima di registrare il tempo*/
    // tempo tra i due eventi in millisecondi
    cudaEventElapsedTime(&elapsed, start, stop);
    
    printf("tempo GPU=%lf\n", elapsed/1000);

    stat = cublasGetVector(N, sizeof(float), d_result, 1, h_result, 1); // ottengo il risultato

    if (N<= 10) {
        printf("\nVettore Prodotto GPU:\n");
        for (int i = 0; i < N; i++) {
            printf("%f\n", h_result[i]);
        }

        printf("\nVettore CPU:\n");
        for (int i = 0; i < N; i++) {
            printf("%f\n", resultCPU[i]);
        }
    }


    /* Rilascio degli eventi */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree (d_matrix);         // Dealloco d_a
    cudaFree (d_vector);         // Dealloco d_b
    cudaFree (d_result);    // Dealloco d_result
    
    cublasDestroy(handle);  // Distruggo l'handle
    
    free(h_matrix);      // Dealloco h_a
    free(h_vector);      // Dealloco h_b
    free(h_result); // Dealloco h_result
    free(resultCPU);  // Dealloco oracolo

    return EXIT_SUCCESS;
}
