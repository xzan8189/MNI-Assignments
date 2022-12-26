/*
    SCOPO
    Prodotto matrice-vettore secondo la strategia a blocchi di colonne

    Compilazione: mpicc ese2.c -o ese2.out -lm
    Esecuzione: mpirun --allow-run-as-root -np 2 ese2.out


*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 

//funzioni ausiliarie
void print_array_int(int arr[], int dim) {
    for (int i=0; i<dim; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

void print_array_double(double arr[], int dim) {
    for (int i=0; i<dim; i++)
        printf("%.2f ", arr[i]);
    printf("\n");
}

void print_matrix_double(double matrix[], int ROW_SIZE, int COL_SIZE) {
    for(int i = 0; i < ROW_SIZE; i++) {
        for(int j = 0; j < COL_SIZE; j++)
            printf("%.2lf\t", matrix[j+(i*COL_SIZE)]);
        printf("\n");
    }
}

/*Genera matrice*/
void generate_matrix(double* A, int ROW_SIZE, int COL_SIZE) {
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

//Esegue il prodotto matrice-vettore
void prod_mat_vett(double w[], double *a, int ROW_SIZE, int COL_SIZE, double v[]) {
    int i, j;
    for(i=0;i<ROW_SIZE;i++) {
        w[i]=0;
        for(j=0;j<COL_SIZE;j++) { 
            w[i] += a[j+(i*COL_SIZE)] * v[j];
        } 
    }    
}

//Esegue il prodotto matrice vettore su trasposta
void prod_mat_vett_trasposta(double w[], double *a, int ROW_SIZE, int COL_SIZE, double v[]) {
    int i, j;
    for(i=0;i<COL_SIZE;i++) {
        w[i]=0;
        for(j=0;j<ROW_SIZE;j++) { 
            //printf("M[j=%d][i=%d]=%.2f  *  v[j=%d]=%f\n", j, i, a[i+(j*COL_SIZE)] * v[j] , j, v[j] );
            w[i] += a[i+(j*COL_SIZE)] * v[j];
        } 
    }    
}

//Restituisce la matrice trasposta di una matrice data in input
double* traspose_vmatrix(double* M, int ROW_SIZE, int COL_SIZE){
	double* N = malloc( sizeof(double) * ROW_SIZE * COL_SIZE );
	for(int i = 0; i < ROW_SIZE; i++){
		for(int j = 0; j < COL_SIZE; j++){
			N[(j * ROW_SIZE) + i] = M[j+(i * COL_SIZE)];
		}
	}
	return N;
}

int main(int argc, char **argv) {
    int nproc;              // Numero di processi totale
    int me;                 // Il mio id
    int m,n;                // Dimensione della matrice
    int local_n;            // Dimensione dei dati locali
    int i,j;                // Iteratori vari 
	double T_inizio, T_fine, T_max;

    // Variabili di lavoro
    double *A;                      //matrice GLOBALE
    double *localA;                 //matrice LOCALE
    double *v;                      //vettore GLOBALE
    double *local_v;                //vettore LOCALE
    double *local_w;                //prodotto matrice-vettore locale
    double *w;                      //prodotto matrice-vettore GLOBALE, ottenuta riunendo tutti prodotti matrice-vettore LOCALI

    /*Attiva MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank (MPI_COMM_WORLD, &me);

    // Se sono a radice
    if(me == 0) {
        printf("inserire numero di righe m= "); 
        fflush(stdout);
        scanf("%d",&m); 
        
        printf("inserire numero di colonne n= "); 
        fflush(stdout);
        scanf("%d",&n);
        // Numero di righe da processare
        local_n = n/nproc;  
        
        // Alloco spazio di memoria
        A = malloc(m * n * sizeof(double));
        v = malloc(sizeof(double)*n);   /*colonne*/
        w =  malloc(sizeof(double)*m);  /*righe*/
        
        /*Stampa del vettore*/
        for (j=0;j<n;j++) {
            v[j]=j; 
        }
        printf("v = ");     
        print_array_double(v, n);
        printf("\n\n");
        
        /*Genero matrice iniziale*/
        generate_matrix(A, m, n);
        if (m<=10 && n<=10) {
            printf("A = \n"); 
            print_matrix_double(A, m, n);
            printf("\n");
        }
        
        /*Calcolo la trasposta*/
        A = traspose_vmatrix(A, m, n);      //nxm
        if (m<=10 && n<=10) {
            printf("A_trasposta= \n"); 
            print_matrix_double(A, m, n);
            printf("\n");
        }
    } // fine me==0


    // Spedisco m, n, local_n e v
    MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD);  
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);            
    MPI_Bcast(&local_n,1,MPI_INT,0,MPI_COMM_WORLD);            

    // Se non sono P0 alloco v, poiché solo il padre l'ha fatto
    local_v = malloc(sizeof(double)*local_n);
        
	MPI_Scatter(
		&v[0], local_n, MPI_DOUBLE,
		&local_v[0], local_n, MPI_DOUBLE,
		0, MPI_COMM_WORLD); 
    printf("\n[%d] local_v ricevuto: ", me);
    print_array_double(local_v, local_n);
    printf("\n");

    // tutti allocano A locale e il vettore dei risultati
    localA = malloc(local_n * m * sizeof(double));
    local_w = malloc(m * sizeof(double));

    // Adesso 0 invia a tutti un pezzo della matrice
    int num = m*local_n;
    printf("num: %d\n", num);
    MPI_Scatter(&A[0], num, MPI_DOUBLE, &localA[0], num, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scriviamo la matrice locale ricevuta
    printf("[%d] localA = \n", me); 
    print_matrix_double(localA, local_n, n);

    // Effettuiamo i calcoli
	T_inizio=MPI_Wtime(); //inizio del cronometro per il calcolo del tempo di inizio
    prod_mat_vett_trasposta(local_w, localA, local_n, m, local_v);
        
    printf("[%d] Local_w: ", me);
    print_array_double(local_w, m);
    printf("\n");

	// P0 raccoglie i risultati parziali, FACENDONE LA SOMMA
	MPI_Reduce(&local_w[0], &w[0], m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	T_fine=MPI_Wtime()-T_inizio; // calcolo del tempo di fine

    // P0 stampa la soluzione
    if(me==0) { 
        if (m<=10 && n<=10) {
            printf("RISULTATO FINALE, w = "); 
            fflush(stdout);
            for(i = 0; i < m; i++)
                printf("%.2f ", w[i]);
            printf("\n");
        }
		printf("\nTempo calcolo locale: %lf ms\n\n", T_fine);
    }

    MPI_Finalize (); /* Disattiva MPI */
    return 0;  
}
