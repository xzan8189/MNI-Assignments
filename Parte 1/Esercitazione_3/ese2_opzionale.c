/**
 *      ! Compilazione: mpicc ese2_opzionale.c -o ese2_opzionale.out -lm
        ! Esecuzione: mpirun --allow-run-as-root -np 4 ese2_opzionale.out
 *
 */
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <math.h>
#include<mpi.h>



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


//Esegue il prodotto matrice vettore su trasposta
void prod_mat_vett(double w[], double *a, int ROW, int COL, double v[]) {
       int i, j;
      for (int i = 0; i < ROW; i++) {
        w[i] = 0;
        for (int j = 0; j < COL; j++) {
            w[i] += a[i * COL + j] * v[j];
        }          
    }     
}

int main(int argc, char *argv[]) {

    int ROW_SIZE, COL_SIZE;
    int my_rank, my_rank2D;
    int nproc;
    int ndim;                           
    int dims[2], periods[2], reorder;   /*caratteristiche della griglia */
    int coords2D[2];
    MPI_Comm comm_grid;                    /* Communicator della griglia */
    int m,n;                // Dimensione della matrice
    int local_m;
    int local_n;             // Dimensione dei dati locali
    int i,j;                // Iteratori vari 
    double T_inizio, T_fine,T_max;

    // Variabili di lavoro
    double *A;                      //matrice
    double *v;                      //vettore
    double *localA;                 //matrice locale
    double *local_TA;               //matrice trasposta locale sulla colonna 0
    double *local_righe;       //blocco di colonne ricevuto dal processore che si trova sulla propria riga e si trova anche in colonna 0
    double *local_w;                //prodotto matrice-vettore locale
    double *w;                      //prodotto matrice-vettore GLOBALE, ottenuta riunendo tutti prodotti matrice-vettore LOCALI
    double *local_y;
    double *local_v;                      //somma dei prodotti matrice-vettore

    /* Tipo blocco */
    MPI_Datatype block,  blockresized;


    /* START MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);    /*identificativo*/
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);      /*numero di processi*/


    ROW_SIZE = 2, COL_SIZE = 2;

     // Se sono radice
    if (my_rank == 0) {
        
        printf("inserire numero di righe (multiplo di ROW_SIZE) m= "); 
        fflush(stdout);
        scanf("%d",&m); 

        printf("inserire numero di colonne (multiplo di COL_SIZE) n= "); 
        fflush(stdout);
        scanf("%d",&n);

        /* Allocazione delle strutture dati */
        A = (double*)malloc(m* n * sizeof(double));
        v = (double*)malloc(n * sizeof(double));
        w = (double*)malloc(m * sizeof(double));
       
       for (j=0;j<n;j++) {
            v[j]=j; 
        }

        /*Genero matrice iniziale*/
        generate_matrix(A, m, n);
        if (m<=10 && n<=10) {
            printf("A = \n"); 
            print_matrix_double(A, m, n);
            printf("\n");
        }

    }
    /* Invio in broadcast del numero di righe e colonne */
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    ndim = 2;   /*dimensione griglia, cioè è 2D*/
 
    /*impostiamo le righe e le colonne della griglia*/

    dims[0] = ROW_SIZE;     /*righe griglia*/
    dims[1] = COL_SIZE;     /*colonne griglia*/ 

    /* Creazione della topologia cartesiana 2D */
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, reorder, &comm_grid);
    MPI_Comm_rank(comm_grid, &my_rank2D);
    MPI_Cart_coords(comm_grid, my_rank2D, ndim, coords2D);

    /* Fase iniziale */
    int belongs[2];
    int id_row, coords1DCol[1], coords1DRow[1];     /*identificatori RIGHE*/
    int id_col;      /*identificatori COLONNE*/
    MPI_Comm comm_row, comm_col;


    /* Creazione comunicatore sottogrigle di colonne 1D */
    belongs[0] = 1; belongs[1] = 0;
    MPI_Cart_sub(comm_grid, belongs, &comm_col);
    MPI_Comm_rank(comm_col, &id_col);
    MPI_Cart_coords(comm_col,id_col, 1, coords1DCol);

    /* Creazione comunicatore sottogriglie di righe 1D */
    belongs[0] = 0; belongs[1] = 1;
    MPI_Cart_sub(comm_grid, belongs, &comm_row);
    MPI_Comm_rank(comm_row, &id_row);
    MPI_Cart_coords(comm_row, id_row, 1, coords1DRow);

    /* Con questa barrier mi assicuro che ogni processore abbia ottenuto le proprie coordinate */
    MPI_Barrier(MPI_COMM_WORLD);
        
    /* Ogni processore calcola le proprie dimensioni locali */
    local_m = m/ dims[0];
    local_n = n / dims[1];

    /* Allocazione strutture dati locali */
    localA = (double*)malloc(local_m * local_n * sizeof(double));
    local_righe = (double*)malloc(local_m * n * sizeof(double));
    local_v = (double*)malloc(local_n * sizeof(double));
    local_w = (double*)malloc(local_m * sizeof(double));
    local_y = (double*)malloc(local_m * sizeof(double));


    ////////// DISTRIBUZIONE MATRICE
    if (coords2D[1] == 0) {/*operazioni da effettuare sulla colonna 0*/
        /*distribuisco le righe lungo la colonna 0 della griglia*/
        MPI_Scatter(A, local_m * n, MPI_DOUBLE, local_righe, local_m * n, MPI_DOUBLE, 0, comm_col);
        
        /* printf("[%d]  localA = \n", my_rank); 
        print_matrix_double(localA, local_n, n);
        printf("\n"); */
    
    }

    //Definiamo il tipo blocco 
    MPI_Type_vector(local_m, local_n, n, MPI_DOUBLE, &block);
    MPI_Type_commit(&block);
    MPI_Type_create_resized(block, 0, local_n*sizeof(double), &blockresized);
    MPI_Type_commit(&blockresized);

    /* Adesso ogni processo lungo la 1 colonna della griglia dovrà distribuire
    il proprio blocco di matrice "local_righe" lungo la propria riga. */
    int num2 = local_m*local_n;     
    MPI_Scatter(local_righe, 1, blockresized, localA, num2 , MPI_DOUBLE, 0, comm_row);


   /////// DISTRIBUZIONE VETTORE
    if (coords2D[0] == 0) {/*operazioni da effettuare sulla riga 0*/ 
        MPI_Scatter(v, local_n, MPI_DOUBLE, local_v, local_n, MPI_DOUBLE, 0, comm_row);
        /* printf("[%d] local_v = ", my_rank);     
        print_array_double(local_v, local_n);
        printf("\n\n"); */
    }

    /* Il primo processore di ogni colonna invia in brodcast, lungo la propria riga di processori, il vettore */
    MPI_Bcast(local_v, local_n, MPI_DOUBLE, 0, comm_col);


    /* Calcolo tempo di inizio */
    MPI_Barrier(MPI_COMM_WORLD);
	T_inizio = MPI_Wtime();//inizio del cronometro per il calcolo del tempo di inizio

    /* Calcolo prodotto mat-vet locale */   
    prod_mat_vett(local_w, localA, local_m, local_n, local_v);

    /* Somme lungo le righe dei prodotti parziali */
    MPI_Reduce(&local_w[0], &local_y[0] , local_m, MPI_DOUBLE, MPI_SUM, 0, comm_row);

    // se sto nella colonna 0
    if (coords2D[1] == 0) {
        MPI_Gather(&local_y[0], local_m, MPI_DOUBLE, &w[0], local_m, MPI_DOUBLE, 0, comm_col);
    }

    /* Sincronizzazione dei processori e calcolo tempo di fine */
    MPI_Barrier(MPI_COMM_WORLD);
	T_fine = MPI_Wtime() - T_inizio;

    /* Calcolo del tempo totale di esecuzione */
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    
    // P0 stampa la soluzione
    if (my_rank == 0) {
         if (m<=10 && n<=10) {
        printf("RISULTATO FINALE, w = "); 
        fflush(stdout);
        for(i = 0; i < m; i++)
            printf("%.2f ", w[i]);
        printf("\n");
        }
        printf("Tempo esecuzione paralleo: %f sec\n", T_max);
    }

    MPI_Finalize();

    

    return 0;
}