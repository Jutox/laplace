/*************************************************
 * Laplace DPC++ para ejecución serial y paralela
 *
 * Temperatura interna es 0.0
 * Valores de temperatura en extremos:
 *
 *      0         T         0
 *   0  +-------------------+  0
 *      |                   |
 *      |                   |
 *      |                   |
 *   T  |                   |  T
 *      |                   |
 *      |                   |
 *      |                   |
 *   0  +-------------------+ 100
 *      0         T        100
 *
 *  Computación Heterogénea, UBB - Primavera 2021
 *  Créditos: John Urbanic, PSC
 *
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <CL/sycl.hpp>
#include "dpc_common.hpp"

using namespace sycl;

// Tamaño de la placa
#define COLUMNS       672
#define ROWS          672     

// Usar 10752 (16 veces más grande) para evaluación de rendimiento

// Cambio de temperatura más grande permitido (Este valor toma 3264 interaciones)
#define MAX_TEMP_ERROR 0.01
// Número máximo de iteraciones
#define MAX_ITERATIONS 4000

double Temperature[ROWS+2][COLUMNS+2];      // Temperatura de la placa
double Temperature_last[ROWS+2][COLUMNS+2]; // Temperatura de la placa última iteración

// Principales rutinas
void initialize();
void track_progress(int iter, double dt);
void laplace_serial(int &iteration, double &dt);
void laplace_parallel(int &iteration, double &dt);


int main(int argc, char *argv[]) {

    int iteration = 1;                                   // iteración actual
    double dt = 100;                                     // cambio más grande en temp
    struct timeval start_time, stop_time, elapsed_time;  // timers
    

    std::cout << "Máximo de iteraciones = " << MAX_ITERATIONS << std::endl;

    std::cout << std::endl << "Ejecución secuencial:" << std::endl;
    initialize();                                        // inicializar Temp_last y condiciones de extremos
    gettimeofday(&start_time,NULL); 
    laplace_serial(iteration, dt);
    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time); 

    std::cout << "Resultados ejecución secuencial" << std::endl;
    std::cout << "Max error en interación " << (iteration - 1) << " fue " << dt << std::endl;
    std::cout << "Tiempo total fue " << (elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0) << " segundos" << std::endl;

    std::cout << std::endl << "Ejecución paralela:" << std::endl;
    iteration=1;                                         
    dt=100;
    initialize();                   
    gettimeofday(&start_time,NULL); 
    laplace_parallel(iteration, dt);
    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time); 

    std::cout << "Resultados ejecución paralela" << std::endl;
    std::cout << "Max error en interación " << (iteration - 1) << " fue " << dt << std::endl;
    std::cout << "Tiempo total fue " << (elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0) << " segundos" << std::endl;

}


// Inicializar Temp_last y condiciones externas
// Temp_last es usada para comenzar la iteración
void initialize(){
    int i,j;

    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }

    // Setear lado izquierdo a 0 y derecho a incremento lineal
    for(i = 0; i <= ROWS+1; i++) {
        Temperature_last[i][0] = 0.0;
        Temperature_last[i][COLUMNS+1] = (100.0/ROWS)*i;
    }
    
    // Setear parte superior a 0 e inferior a incremento lineal
    for(j = 0; j <= COLUMNS+1; j++) {
        Temperature_last[0][j] = 0.0;
        Temperature_last[ROWS+1][j] = (100.0/COLUMNS)*j;
    }
}


// Imprimir diagonal de larte inferior derecha
void track_progress(int iteration, double dt) {
    int i;

    std::cout << "---- Iteración "<< iteration << " dt = " << dt << "----" << std::endl;
    for(i = ROWS-5; i <= ROWS-3; i++) {
        std::cout << "[" << i << "," << i << "]:" << Temperature[i][i] << "   ";
    }
    std::cout << std::endl;
}

// Calcular Laplace en forma secuencial
void laplace_serial(int &iteration, double &dt){
    int i, j;   

    // Iterar hasta alcanzar un estado de estabilización o número máximo de iteraciones
    while ( dt > MAX_TEMP_ERROR && iteration <= MAX_ITERATIONS ) {

        // Cálculo principal:
        // Por cada posición de la matriz calcular el promedio de los vecinos
        for(i = 1; i <= ROWS; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }
        
        dt = 0.0; // resetear cambio de temperatura más grande

        // Copiar matriz y encontrar el máximo cambio de temperatura
        for(i = 1; i <= ROWS; i++){
            for(j = 1; j <= COLUMNS; j++){
	      dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
	      Temperature_last[i][j] = Temperature[i][j];
            }
        }

        // Imprimir valores periodicamente
        if((iteration % 100) == 0) {
	        track_progress(iteration, dt);
        }
	    iteration++;
    }
}

// Calcular Laplace en forma paralela
void laplace_parallel(int &iteration, double &dt){

    /*
     * AGREGUE AQUÍ SU SOLUCIÓN
     */
    int i, j;
    queue q;

    buffer temp_buf(reinterpret_cast <double *> (Temperature), range(ROWS+2,COLUMNS+2));
    buffer temperature_buf(reinterpret_cast<double *>(Temperature_last), range(ROWS+2,COLUMNS+2));
    buffer <double, 1> buffdt (&dt, 1);


    // Iterar hasta alcanzar un estado de estabilización o número máximo de iteraciones
    while ( dt > MAX_TEMP_ERROR && iteration <= MAX_ITERATIONS ) {

        

        // Cálculo principal:
        // Por cada posición de la matriz calcular el promedio de los vecinos
        q.submit([&](handler &h){
            accessor temp_acc(temp_buf,h, write_only);
            accessor temperature_acc(temperature_buf,h, read_only);

            range<2> global(14,14);
            range<2> local(1,1);

            h.parallel_for(nd_range<2>(global,local),[=](nd_item<2> index){

                  int portion = (ROWS)/14;
                  int start_i = index.get_global_id() [0] * portion;
                  int end_i = start_i + portion;
                  int start_j = index.get_global_id() [1] * portion;
                  int end_j = start_j + portion;

                for(int i= start_i; i<= end_i; i++){
                    for(int j= start_j; j <= end_j;j++){
                        temp_acc[i][j] = 0.25 * (temperature_acc[i+1][j] + temperature_acc[i-1][j] + temperature_acc[i][j+1] + temperature_acc[i][j-1]);
                    }
                }
            }); 
         });
        

        dt = 0.0;

       
        q.submit([&](handler &h){
            accessor temp_acc(temp_buf,h, read_only);
            accessor temperature_acc(temperature_buf,h, write_only);

            range<2> global(14,14);
            range<2> local(1,1);

            auto dtmax = reduction(buffdt, h, maximum<>());

            h.parallel_for(nd_range<2>({global,local}), dtmax, [=](nd_item<2>index, auto& max){

                int portion = (ROWS)/14;
                int start_i = index.get_global_id() [0] * portion;
                int end_i = start_i + portion;
                int start_j = index.get_global_id() [1] * portion;
                int end_j = start_j + portion;
   
                for(int i= start_i ; i <= end_i; i++){
                    for(int j= start_j; j <= end_j;j++){
                        max.combine(fabs( temp_acc[i][j]-temperature_acc[i][j]));
                        temperature_acc[i][j] = temp_acc[i][j];
                    }
                }
            }); 
        });
       
        host_accessor buffdt_host(buffdt);


        // Imprimir valores periodicamente
        if((iteration % 100) == 0) {
	        track_progress(iteration, dt);
        }
	    iteration++;
    }
}