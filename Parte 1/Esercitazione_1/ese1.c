/*
Sviluppare e implementare in linguaggio C un algoritmo seriale per il calcolo
della somma di n numeri reali.

! Compilazione: mpicc ese1.c -o ese1.out
! Esecuzione: mpirun --allow-run-as-root -np 1 ese1.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

//Metodi ausiliari
void print_array_double(double arr[], int dim) {
    for (int i=0; i<dim; i++)
        printf("%.2f ", arr[i]);
    printf("\n");
}

double somma_array(double arr[], int dim) {
	double somma = 0;
    for (int i=0; i<dim; i++)
        somma += arr[i];
        
    return somma;
}

int main(int argc, char *argv[]){
	int n, my_rank;
	double *vett_loc;
	double T_inizio, T_fine;
	
	/* MPI INIT */
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


	if (!my_rank) {
		// input
		printf("Inserire il numero di elementi da sommare: ");
		fflush(stdout);
		scanf("%d",&n);
		vett_loc=(double*)calloc(n,sizeof(double));

		/*Inizializza la generazione random degli addendi utilizzando l'ora attuale del sistema*/                
		srand((unsigned int) time(0)); 
		
		/* creazione del vettore contenente numeri casuali */
		for(int i=0; i<n; i++) {
		
			int min = -2, max = 2;
			double range = (max - min);
			double div = RAND_MAX / range;
			
			vett_loc[i]= min + (rand()/div);
		}

		//Calcolo la somma
		T_inizio=MPI_Wtime();
		double somma = somma_array(vett_loc, n);
		
		T_fine=MPI_Wtime()-T_inizio;

		printf("La somma dei %d numeri reali è: %.2f\n", n, somma);
		printf("\nTempo calcolo locale: %lf secondi\n\n", T_fine);
	}

	MPI_Finalize();
	return 0;
}






















