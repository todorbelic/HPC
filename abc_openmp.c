#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h>
#include <string.h>

#define DIM 1500
#define FOOD_NUMBER 4000
#define MAX_ITER 50000

#define LOWER_BOUND -100
#define UPPER_BOUND 100
#define LIMIT 100

typedef struct {
    double *position;
    double fitness;
    int trial;
} FoodSource;

FoodSource *foods;
double *global_best;
double global_best_fitness = DBL_MAX;

FILE *convergence_log = NULL;

double sphere_function(double *x) {
    double sum = 0.0;
    for (int i = 0; i < DIM; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

double objective_function(double *x) {
    return sphere_function(x);
}

double random_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

double calculate_fitness(double *solution) {
    double obj_value = objective_function(solution);
    return 1.0 / (1.0 + obj_value);
}

void log_convergence(int iteration, double elapsed_time) {
    if (convergence_log != NULL) {
        fprintf(convergence_log, "%d,%.6f,%.10e\n", 
                iteration, elapsed_time, global_best_fitness);
        fflush(convergence_log);
    }
}

void initialize_population() {

    #pragma omp parallel for
    for (int i = 0; i < FOOD_NUMBER; i++) {
        unsigned int seed = time(NULL) + i;
        #ifdef USE_OPENMP
        seed += omp_get_thread_num() * 1000;
        #endif
        
        for (int j = 0; j < DIM; j++) {
            foods[i].position[j] = random_double(LOWER_BOUND, UPPER_BOUND);
        }
        
        foods[i].fitness = calculate_fitness(foods[i].position);
        foods[i].trial = 0;

        double obj_value = objective_function(foods[i].position);

        #pragma omp critical
        {
            if (obj_value < global_best_fitness) {
                global_best_fitness = obj_value;
                for (int j = 0; j < DIM; j++) {
                    global_best[j] = foods[i].position[j];
                }
            }
        }
    }
}

void employed_bee_phase() {

    #pragma omp parallel for
    for (int i = 0; i < FOOD_NUMBER; i++) {
        unsigned int seed = time(NULL) + i;
        #ifdef USE_OPENMP
        seed += omp_get_thread_num() * 1000;
        #endif
        
        double *new_solution = (double*)malloc(DIM * sizeof(double));
        
        for (int j = 0; j < DIM; j++) {
            new_solution[j] = foods[i].position[j];
        }

        int dim_to_modify = rand() % DIM;
        int bee_friend = rand() % FOOD_NUMBER;
        
        while (bee_friend == i) {
            bee_friend = rand() % FOOD_NUMBER;
        }

        double phi = random_double(-1.0, 1.0);
        new_solution[dim_to_modify] = foods[i].position[dim_to_modify] +
                                    phi * (foods[i].position[dim_to_modify] - 
                                          foods[bee_friend].position[dim_to_modify]);

        if (new_solution[dim_to_modify] < LOWER_BOUND) 
            new_solution[dim_to_modify] = LOWER_BOUND;
        if (new_solution[dim_to_modify] > UPPER_BOUND) 
            new_solution[dim_to_modify] = UPPER_BOUND;

        double new_fitness = calculate_fitness(new_solution);

        if (new_fitness > foods[i].fitness) {
            foods[i].fitness = new_fitness;
            foods[i].position[dim_to_modify] = new_solution[dim_to_modify];
            foods[i].trial = 0;
            
            double obj_value = objective_function(new_solution);
            
            #pragma omp critical
            {
                if (obj_value < global_best_fitness) {
                    global_best_fitness = obj_value;
                    for (int j = 0; j < DIM; j++) {
                        global_best[j] = new_solution[j];
                    }
                }
            }
        } else {
            foods[i].trial++;
        }
        
        free(new_solution);
    }
}

void onlooker_bee_phase() {
    double total_fitness = 0.0;
    
    #pragma omp parallel for reduction(+: total_fitness)
    for (int i = 0; i < FOOD_NUMBER; i++) {
        total_fitness += foods[i].fitness;
    }

    #pragma omp parallel for
    for (int i = 0; i < FOOD_NUMBER; i++) {
        unsigned int seed = time(NULL) + i;
        #ifdef USE_OPENMP
        seed += omp_get_thread_num() * 2000;
        #endif
        
        double prob = foods[i].fitness / total_fitness;

        if (((double)rand_r(&seed) / RAND_MAX) < prob) {
            double *new_solution = (double*)malloc(DIM * sizeof(double));

            for (int j = 0; j < DIM; j++) {
                new_solution[j] = foods[i].position[j];
            }

            int dim_to_modify = rand() % DIM;
            int bee_friend = rand() % FOOD_NUMBER;
            
            while (bee_friend == i) {
                bee_friend = rand() % FOOD_NUMBER;
            }

            double phi = random_double(-1.0, 1.0);
            new_solution[dim_to_modify] = foods[i].position[dim_to_modify] +
                                        phi * (foods[i].position[dim_to_modify] - 
                                              foods[bee_friend].position[dim_to_modify]);
            
            if (new_solution[dim_to_modify] < LOWER_BOUND) 
                new_solution[dim_to_modify] = LOWER_BOUND;
            if (new_solution[dim_to_modify] > UPPER_BOUND) 
                new_solution[dim_to_modify] = UPPER_BOUND;

            double new_fitness = calculate_fitness(new_solution);

            if (new_fitness > foods[i].fitness) {
                foods[i].position[dim_to_modify] = new_solution[dim_to_modify];
                foods[i].fitness = new_fitness;
                foods[i].trial = 0;

                double obj_value = objective_function(new_solution);

                #pragma omp critical
                {
                    if (obj_value < global_best_fitness) {
                        global_best_fitness = obj_value;
                        for (int j = 0; j < DIM; j++) {
                            global_best[j] = new_solution[j];
                        }
                    }
                }
            } else {
                foods[i].trial++;
            }
            
            free(new_solution);
        }
    }
}

void scout_bee_phase() {

    #pragma omp parallel for
    for (int i = 0; i < FOOD_NUMBER; i++) {
        if (foods[i].trial >= LIMIT) {
            unsigned int seed = time(NULL) + i;
            #ifdef USE_OPENMP
            seed += omp_get_thread_num() * 3000;
            #endif
            
            for (int j = 0; j < DIM; j++) {
                foods[i].position[j] = random_double(LOWER_BOUND, UPPER_BOUND);
            }

            foods[i].fitness = calculate_fitness(foods[i].position);
            foods[i].trial = 0;

            double obj_value = objective_function(foods[i].position);
            
            #pragma omp critical
            {
                if (obj_value < global_best_fitness) {
                    global_best_fitness = obj_value;
                    for (int j = 0; j < DIM; j++) {
                        global_best[j] = foods[i].position[j];
                    }
                }
            }
        }
    }
}

void print_usage(char *prog_name) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -d, --dim DIM              Problem dimension (default: 150)\n");
    printf("  -f, --food FOOD            Number of food sources (default: 1000)\n");
    printf("  -i, --iter ITER            Maximum iterations (default: 5000)\n");
    printf("  -l, --log FILE             Convergence log file (default: convergence.csv)\n");
    printf("  -h, --help                 Show this help message\n");
    printf("\nExample:\n");
    printf("  %s -d 100 -f 500 -i 2000 -l results.csv\n", prog_name);
}

int main(int argc, char *argv[]) {
    char log_filename[256] = "convergence.csv";
    
    foods = (FoodSource*)malloc(FOOD_NUMBER * sizeof(FoodSource));
    global_best = (double*)malloc(DIM * sizeof(double));
    
    for (int i = 0; i < FOOD_NUMBER; i++) {
        foods[i].position = (double*)malloc(DIM * sizeof(double));
    }
    
    srand(time(NULL));
    
    #ifdef USE_OPENMP
    printf("OpenMP Artificial Bee Colony Algorithm\n");
    #else
    printf("Sequential Artificial Bee Colony Algorithm\n");
    #endif
    printf("=========================================\n");
    printf("Problem Dimension: %d\n", DIM);
    printf("Food Sources: %d\n", FOOD_NUMBER);
    printf("Max Iterations: %d\n", MAX_ITER);
    #ifdef USE_OPENMP
    printf("OpenMP Threads: %d\n", omp_get_max_threads());
    #endif
    printf("Convergence Log: %s\n\n", log_filename);
    
    convergence_log = fopen(log_filename, "w");
    if (convergence_log == NULL) {
        printf("Warning: Could not open log file %s\n", log_filename);
    } else {
        fprintf(convergence_log, "Iteration,Time,Fitness\n");
    }
    
    #ifdef USE_OPENMP
    double start_time = omp_get_wtime();
    #else
    clock_t start_clock = clock();
    #endif
    
    initialize_population();
    printf("Initial Best Fitness: %.10e\n\n", global_best_fitness);
    
    log_convergence(0, 0.0);
    
    for (int iter = 1; iter <= MAX_ITER; iter++) {
        if (global_best_fitness < 1e-2) { 
            break;
        }
        employed_bee_phase();
        onlooker_bee_phase();
        scout_bee_phase();
        
        #ifdef USE_OPENMP
        double elapsed = omp_get_wtime() - start_time;
        #else
        double elapsed = (double)(clock() - start_clock) / CLOCKS_PER_SEC;
        #endif
        
        if (iter % 10 == 0) {
            log_convergence(iter, elapsed);
        }
        
        if (iter % 100 == 0) {
            printf("Iteration %d: Best Fitness = %.10e, Time = %.4fs\n", 
                   iter, global_best_fitness, elapsed);
        }
    }
    
    #ifdef USE_OPENMP
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;
    #else
    clock_t end_clock = clock();
    double total_time = (double)(end_clock - start_clock) / CLOCKS_PER_SEC;
    #endif
    
    log_convergence(MAX_ITER, total_time);
    
    printf("\n=========================================\n");
    printf("Optimization Complete!\n");
    printf("Final Best Fitness: %.10e\n", global_best_fitness);
    printf("Execution Time: %.4f seconds\n", total_time);
    printf("\nBest Solution (first 10 dimensions):\n");
    for (int i = 0; i < (DIM < 10 ? DIM : 10); i++) {
        printf("x[%d] = %.6f\n", i, global_best[i]);
    }
    
    if (convergence_log != NULL) {
        fclose(convergence_log);
    }
    
    for (int i = 0; i < FOOD_NUMBER; i++) {
        free(foods[i].position);
    }
    free(foods);
    free(global_best);
    
    return 0;
}