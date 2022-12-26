/*
    SCOPO
    Prodotto matrice-vettore sequenziale
    Compilazione: mpicc sequenziale.c -o sequenziale.out -lm
    Esecuzione: mpirun --allow-run-as-root -np 1 sequenziale.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 

//funzioni ausiliarie
void print_matrix_double(double matrix[], int ROW_SIZE, int COL_SIZE) {
    for(int i = 0; i < ROW_SIZE; i++) {
        for(int j = 0; j < COL_SIZE; j++)
            printf("%.2lf\t", matrix[j+(i*COL_SIZE)]);
        printf("\n");
    }
}

void print_array_double(double arr[], int dim) {
    for (int i=0; i<dim; i++)
        printf("%.2f ", arr[i]);
    printf("\n");
    
}


//Funzione che esegue il prodotto matrice vettore
void prod_mat_vett(double w[], double *a, int ROWS, int COLS, double v[]) {
    int i, j;
    for(i=0;i<ROWS;i++) {
        w[i]=0;
        for(j=0;j<COLS;j++) { 
            w[i] += a[i*COLS+j]* v[j];
        } 
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


int main(int argc, char **argv) {
    int nproc;              // Numero di processi totale
    int me;                 // Il mio id
    int m,n;                // Dimensione della matrice
    int i,j;                // Iteratori vari 
	double T_inizio, T_fine, T_max;

    // Variabili di lavoro
    double *A, *v, *w;

    /*Attiva MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &me);

    // Se sono a radice
    if(me == 0) {
        printf("inserire numero di righe m = \n"); 
        scanf("%d",&m);
        
        printf("inserire numero di colonne n = \n"); 
        scanf("%d",&n);
        printf("\n\n");
    
        // Alloco spazio di memoria
        A = malloc(m * n * sizeof(double)); /*matrice*/
        v = malloc(sizeof(double)*n);       /*vettore*/
        w = malloc(sizeof(double)*m);       /*vettore risultante*/
        
        //Genero il vettore
        for (j=0;j<n;j++) {
            v[j]=j;  
        }

        if (m<=10 && n<=10) {
            printf("Vettore v = \n");     
            print_array_double(v, n);
            printf("\n\n");
         }

       /*Genero matrice iniziale*/
       generate_matrix(A, m, n);
       if (m<=10 && n<=10) {
            printf("Matrice A = \n"); 
            print_matrix_double(A, m, n);
            printf("\n\n");
        }
    } // fine me==0

    //printf("Iniziato a contare\n");
	T_inizio=MPI_Wtime(); //inizio del cronometro per il calcolo del tempo di inizio

    // Effettuiamo i calcoli
    prod_mat_vett(w,A,m,n,v);

	T_fine=MPI_Wtime()-T_inizio; // calcolo del tempo di fine

    // 0 stampa la soluzione
    if(me==0) { 
        if (m<=10 && n<=10) {
            printf("RISULTATO FINALE, w =\n"); 
            print_array_double(w, m);
        }
		printf("\nTempo calcolo locale: %lf s\n\n", T_fine);
    }

    MPI_Finalize (); /* Disattiva MPI */
    return 0;  
}