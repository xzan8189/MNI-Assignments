/*
    SCOPO:
    
    ? ATTENZIONE: la matrice è di dimensione 3x3, quindi devi eseguire con 9 processori poiché 3x3=9
    
    ! Compilazione: mpicc ese1.c -o ese1.out -lm
    ! Esecuzione: mpirun --allow-run-as-root -np 4 ese1.out
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



//Esegue il prodotto matrice vettore su trasposta
void prod_mat_vett_trasposta(double w[], double *a, int ROW, int COL, double v[]) {
       int i, j;
    for(i=0;i<COL;i++) {
        w[i]=0;
        for(j=0;j<ROW;j++) { 
            //printf("M[j=%d][i=%d]=%.2f  *  v[j=%d]=%f\n", j, i, a[i+(j*COL_SIZE)] * v[j] , j, v[j] );
            w[i] += a[i+(j*COL)] * v[j];
        } 
    }      
}

void main(int argc, char *argv[]) {
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
    double *local_TA_colonne;       //blocco di colonne ricevuto dal processore che si trova sulla propria riga e si trova anche in colonna 0
    double *local_w;                //prodotto matrice-vettore locale
    double *w;                      //prodotto matrice-vettore GLOBALE, ottenuta riunendo tutti prodotti matrice-vettore LOCALI
    double *y;                      //somma dei prodotti matrice-vettore


    /* START MPI */
    MPI_Init(&argc, &argv);  
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);    /*identificativo*/
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);      /*numero di processi*/

    ROW_SIZE = 2, COL_SIZE = 2;

    /*controllo*/
    if (ROW_SIZE*COL_SIZE != nproc) {
        if (my_rank==0) {
            printf("ERRORE: dimensione griglia è %dx%d ed nproc è %d, ma nproc dovrebbe essere '%d'\n", ROW_SIZE, COL_SIZE, nproc, ROW_SIZE*COL_SIZE);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }


    // Se sono radice
    if (my_rank == 0) {
        printf("inserire numero di righe (multiplo di ROW_SIZE) m= "); 
        fflush(stdout);
        scanf("%d",&m); 
        
        printf("inserire numero di colonne (multiplo di COL_SIZE) n= "); 
        fflush(stdout);
        scanf("%d",&n);
        // Numero di righe da processare
        //local_m = m/nproc;  
        
        // Alloco spazio di memoria
        A = malloc(m * n * sizeof(double));
        v = malloc(n*sizeof(double));   /*colonne*/
        w =  malloc(m*sizeof(double));  
        
        for (j=0;j<n;j++) {
            v[j]=j; 
        }
        
         /*Stampa del vettore*/
        /*
        printf("v = ");     
        print_array_double(v, n);
        printf("\n\n");*/
        
        /*Genero matrice iniziale*/
        generate_matrix(A, m, n);
        if (m<=10 && n<=10) {
            printf("A = \n"); 
            print_matrix_double(A, m, n);
            printf("\n");
        }
    }

	MPI_Barrier(MPI_COMM_WORLD);

	/////////// CREAZIONE GRIGLIA PROCESSORI
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ROW_SIZE, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&COL_SIZE, 1, MPI_INT, 0, MPI_COMM_WORLD);

    ndim = 2;   /*dimensione griglia, cioè è 2D*/
 
    /*impostiamo le righe e le colonne della griglia*/

    dims[0] = ROW_SIZE;     /*righe griglia*/
    dims[1] = COL_SIZE;     /*colonne griglia*/ 

    /*importiamo la periodicità*/
    periods[0] = 0;
    periods[1] = 0; 

    /*non riordino gli id dei processori*/
    reorder = 1;

    /*Creazione griglia*/
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, reorder, &comm_grid);
    MPI_Comm_rank(comm_grid, &my_rank2D); 
    MPI_Cart_coords(comm_grid, my_rank2D, ndim, coords2D);

    ///////////// Creazione delle sottogriglie delle righe e delle colonne
    /* Fase iniziale */
    int belongs[2];
    int id_row, coord_row;      /*identificatori RIGHE*/
    int id_col, coord_col;      /*identificatori COLONNE*/
    MPI_Comm comm_row, comm_col;

	/* Creazione sottogriglie di righe */
	belongs[0] = 0;
	belongs[1] = 1;
	MPI_Cart_sub(comm_grid, belongs, &comm_row);
	MPI_Comm_rank(comm_row, &id_row);
	MPI_Cart_coords(comm_row, id_row, ndim, &coord_row);

	/* Creazione sottogriglie di colonne */  
	belongs[0] = 1;
	belongs[1] = 0;
	MPI_Cart_sub(comm_grid, belongs, &comm_col);
	MPI_Comm_rank(comm_col, &id_col);
	MPI_Cart_coords(comm_col, id_col, ndim, &coord_col);

    /* printf("[%d] coords2D: (%d, %d)\t coord_row: %d\t coord_col: %d\n", 
            my_rank, coords2D[0], coords2D[1], coord_row, coord_col); */

 /* Con questa barrier mi assicuro che ogni processore abbia ottenuto le proprie coordinate */
    MPI_Barrier(MPI_COMM_WORLD);

    ////////// DISTRIBUZIONE MATRICE
    local_m = m/ROW_SIZE;
    local_n = n/COL_SIZE;

    // tutti allocano A locale e il vettore dei risultati
    localA = malloc(local_m * n * sizeof(double));
    local_TA = NULL;

    if (coord_row == 0) { /*operazioni da effettuare sulla colonna 0*/ //puoi mettere anche "coords2D[1] == 0" nell'if
        int num = local_m*n;

        /*distribuisco le righe lungo la colonna 0 della griglia*/
		MPI_Scatter(A, num, MPI_DOUBLE, localA, num, MPI_DOUBLE, 0, comm_col);
        /* printf("[%d] localA = \n", my_rank); 
        print_matrix_double(localA, local_n, n);
        printf("\n"); */

        /*calcolo la trasposta di localA*/
        local_TA = traspose_vmatrix(localA, local_m, n);
        /* printf("[%d] local_TA trasposta= \n", my_rank); 
        print_matrix_double(local_TA, local_n, n);
        printf("\n"); */
    }

    /* Adesso ogni processo lungo la 1 colonna della griglia dovrà distribuire
    il proprio blocco di matrice "local_TA" lungo la propria riga.
    Visto che si tratta di una trasposta allora verranno distribuite colonne. */
    int num2 = local_m*local_n;     /*quante colonne dobbiamo inviare a ognuno*/
    local_TA_colonne = malloc(local_m * local_n * sizeof(double));
	MPI_Scatter(local_TA, num2, MPI_DOUBLE, local_TA_colonne, num2, MPI_DOUBLE, 0, comm_row);


    printf("[%d] local_TA_colonne = \n", my_rank); 
    print_matrix_double(local_TA_colonne, local_m, local_n);
    printf("\n");


    /////// DISTRIBUZIONE VETTORE
    
	double* local_v = malloc(local_n * sizeof(double));
    if (coord_col == 0) {   /*operazioni da effettuare sulla riga 0*/ //puoi mettere anche "coords2D[0] == 0" nell'if
		MPI_Scatter(v, local_n, MPI_DOUBLE, local_v, local_n, MPI_DOUBLE, 0, comm_row);
        /* printf("[%d] local_v = ", my_rank);     
        print_array_double(local_v, local_n);
        printf("\n\n"); */
    }

	// Distribuire la stessa porzione di vettore lungo la propria colonna
	MPI_Bcast(local_v, local_n, MPI_DOUBLE, 0, comm_col);

    /*printf("[%d] local_v = ", my_rank);     
    print_array_double(local_v, local_n);
    printf("\n\n"); 
    */ 
  


    MPI_Finalize();
}