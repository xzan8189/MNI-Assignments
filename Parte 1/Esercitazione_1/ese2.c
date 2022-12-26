/* --------------------------------------------------------------------------
                                    II STRATEGIA

	SCOPO
	Sviluppare e implementare in linguaggio C--MPI un algoritmo parallelo 
	per  il calcolo della somma di n numeri reali, che utilizzi la 
	II strategia di parallelizzazione.

    ! Compilazione: mpicc ese2.c -o ese2.out
    ! Esecuzione: mpirun --allow-run-as-root -np 2 ese2.out 10000000
--------------------------------------------------------------------------
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

void print_array_int(int arr[], int dim) {
    for (int i=0; i<dim; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

//Distribuisce in maniera equa (se possibile) le righe di una matrice ai diversi processori
void countsPerProcess(int send_counts[], int nprocs, int divisione_intera, int resto) {
    for (int i = 0; i < nprocs; i++) {
        send_counts[i] = (i < resto) ? divisione_intera + 1 : divisione_intera;
    }
}

int main (int argc, char **argv) {
	/*dichiarazioni variabili*/
    int my_rank, nprocs, tag;
	int n, counts, i, resto, divisione_intera;
	int ind, p, r, sendTo, recvBy;
	//int *vett, *my_vett;
	double *vett, *my_vett;
	int *potenze, passi=0;
	double sommaloc=0, tmp;
	double T_inizio, T_fine, T_max;

    //Dati utili per la suddivisione in maniera equa (se possibile) della matrice
    int *send_counts;

	MPI_Status info;
	
	/*Inizializzazione dell'ambiente di calcolo MPI*/
	MPI_Init(&argc,&argv);
	/*assegnazione IdProcessore a my_rank*/
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/*assegna numero processori a nprocs*/
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* lettura e inserimento dati*/
	if (my_rank==0) {
		printf("Inserire il numero di elementi da sommare: ");
		fflush(stdout);
		//scanf("%d",&n);
		n = atoi(argv[1]);
		printf("%d\n", n);
		
   		vett=(double*)calloc(n,sizeof(double));
        //send_counts = malloc(nprocs * sizeof(int)); //Array che conterrà il num. di elementi di ogni processo
	}

	/*invio del valore di n a tutti i processori appartenenti a MPI_COMM_WORLD*/
	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

    /*numero di addendi da assegnare a ciascun processore*/
	divisione_intera = n / nprocs; // divisione intera
    resto = n % nprocs;            // resto della divisione

	/* Calcolo il numero di elementi che deve avere ogni processore */
	//countsPerProcess(send_counts, nprocs)
	if (my_rank<resto) {
		counts=divisione_intera+1;
	} else {
		counts=divisione_intera;
	}
	
    /*allocazione di memoria del vettore per le somme parziali */
	my_vett=(double*)calloc(counts, sizeof(double));

	// P0 genera e assegna gli elementi da sommare
	if (my_rank==0) {
        /*Inizializza la generazione random degli addendi utilizzando l'ora attuale del sistema*/                
        srand((unsigned int) time(0));                
		
        for(i=0; i<n; i++) {
			/*creazione del vettore contenente numeri casuali */
			int min = -2, max = 2;
			double range = (max - min);
			double div = RAND_MAX / range;
			
			vett[i]= min + (rand()/div);
			//*(vett+i)=(int)rand()%5-2;
		}
		
   		// Stampa del vettore che contiene i dati da sommare se e solo se sono meno di 100 
		if (n<100) {
			for (i=0; i<n; i++) {
				printf("vett[%d] = %f\n", i, vett[i]);
			}
        }

	    // assegnazione dei primi addendi a P0
        for(i=0;i<counts;i++) {
			my_vett[i]=vett[i];
		}
  
  	    // ind è il numero di elementi già assegnati     
        ind=counts;
        
		/* P0 assegna i restanti elementi agli altri processori */
		for(i=1; i<nprocs; i++) {
			tag=i; /* tag del messaggio uguale all'id del processo che riceve*/
			/*SE ci sono elementi in sovrannumero da ripartire tra i processori*/
            if (i<resto) {
				/*il processore P0 gli invia il corrispondente vettore locale considerando un elemento in più*/
				MPI_Send(vett+ind,counts,MPI_DOUBLE,i,tag,MPI_COMM_WORLD);
				ind += counts;
			} else {
				/*il processore P0 gli invia il corrispondente vettore locale*/
				MPI_Send(vett+ind,divisione_intera,MPI_DOUBLE,i,tag,MPI_COMM_WORLD);
				ind += divisione_intera;
			}// end if
		}//end for
	}
	/*SE non siamo il processore P0 riceviamo i dati trasmessi dal processore P0*/
    else {
		// tag è uguale numero di processore che riceve
		tag=my_rank;
  
		/*fase di ricezione*/
		MPI_Recv(my_vett,counts,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&info);
	}// end if

   
	
	/* sincronizzazione dei processori del contesto MPI_COMM_WORLD*/
	MPI_Barrier(MPI_COMM_WORLD);
 
	T_inizio=MPI_Wtime(); //inizio del cronometro per il calcolo del tempo di inizio

	for(i=0;i<counts;i++) {
		/*ogni processore effettua la somma parziale*/
		sommaloc=sommaloc+*(my_vett+i);
	}

	//  calcolo di p=log_2 (nprocs)
	p=nprocs;

	while(p!=1) {
		/*shifta di un bit a destra*/
		p=p>>1;
		passi++;
	}
 
	/* creazione del vettore potenze, che contiene le potenze di 2*/
	potenze=(int*)calloc(passi+1,sizeof(int));
		
	for(i=0;i<=passi;i++) {
		potenze[i] = p<<i;
	}

    // INIZIO DELLA 2 STRATEGIA
	/* calcolo delle somme parziali e combinazione dei risultati parziali */
	for(i=0;i<passi;i++) {
		// ... calcolo identificativo del processore (calcoliamo il resto)
		r = my_rank % potenze[i+1];
		
		// Se il resto è uguale a 2^i, il processore my_rank invia
		if(r==potenze[i]) {
			// calcolo dell'id del processore a cui spedire la somma locale
			sendTo = my_rank - potenze[i];
			tag=sendTo;
			MPI_Send(&sommaloc,1,MPI_INT,sendTo,tag,MPI_COMM_WORLD);
		}
		else if(r==0) // il processore my_rank riceve
		{
			recvBy = my_rank + potenze[i];
			tag=my_rank;
            //printf("my_rank: %d, recvBy: %d\n", my_rank, recvBy);
			MPI_Recv(&tmp,1,MPI_INT,recvBy,tag,MPI_COMM_WORLD,&info);

			// calcolo della somma parziale al passo i (solo quando ricevo, perché è la 2 strategia)
			sommaloc=sommaloc+tmp;
		}//end
	}// end for

	MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
	T_fine=MPI_Wtime()-T_inizio; // calcolo del tempo di fine
 
	/* calcolo del tempo totale di esecuzione*/
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

	/*stampa a video dei risultati finali*/
	if(my_rank==0) {
		printf("\nProcessori impegnati: %d\n", nprocs);
		printf("\nLa somma e': %f\n", sommaloc);
		printf("\nTempo calcolo locale: %lf secondi\n", T_fine);
		printf("\nMPI_Reduce max time: %f\n",T_max);
	}// end if
 
	/*routine chiusura ambiente MPI*/
	MPI_Finalize();
}// fine programma