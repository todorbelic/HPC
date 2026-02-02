#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <mpi.h>

#define DIM 1500
#define FOOD_NUMBER 3000
#define MAX_ITER 60000

#define LOWER_BOUND -100.0
#define UPPER_BOUND 100.0
#define LIMIT 100

typedef struct {
    double position[DIM];   
    double fitness;         
    int trial;              
} FoodSource;

FoodSource *local_foods;
int local_food_count;
double global_best[DIM];
double global_best_fitness = DBL_MAX;
double local_best_fitness = DBL_MAX;

FILE *convergence_log = NULL;

int rank, size;

double sphere_function(double *x);
double objective_function(double *x);
double calculate_fitness(double *solution);
void initialize_population();
void employed_bee_phase();
void onlooker_bee_phase();
void scout_bee_phase();
void communicate_best_solution();
void gather_all_solutions(FoodSource **all_foods);
double random_double(double min, double max);
void print_results(int iteration);
void log_convergence(int iteration, double elapsed_time);
void print_usage(char *prog_name);

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
    if (obj_value < 1e-10) {
        // log(1 + x) ≈ x za male x, ali zadržava preciznost
        return 1.0 / (1.0 + log1p(obj_value / 1e-10));
    } else {
        return 1.0 / (1.0 + obj_value);
    }
}

void log_convergence(int iteration, double elapsed_time) {
    if (convergence_log != NULL && rank == 0) {
        fprintf(convergence_log, "%d,%.6f,%.10e\n", 
                iteration, elapsed_time, global_best_fitness);
        fflush(convergence_log);
    }
}

void initialize_population() {
    for (int i = 0; i < local_food_count; i++) {
        for (int j = 0; j < DIM; j++) {
            local_foods[i].position[j] = random_double(LOWER_BOUND, UPPER_BOUND);
        }
        
        local_foods[i].fitness = calculate_fitness(local_foods[i].position);
        local_foods[i].trial = 0;
        
        double obj_value = objective_function(local_foods[i].position);
        if (obj_value < local_best_fitness) {
            local_best_fitness = obj_value;
        }
    }
    
    communicate_best_solution();
}

void communicate_best_solution() {
    double local_best[DIM];
    local_best_fitness = DBL_MAX;
    
    for (int i = 0; i < local_food_count; i++) {
        double obj_value = objective_function(local_foods[i].position);
        if (obj_value < local_best_fitness) {
            local_best_fitness = obj_value;
            for (int j = 0; j < DIM; j++) {
                local_best[j] = local_foods[i].position[j];
            }
        }
    }
    
    struct {
        double fitness;
        int rank;
    } local_data, global_data;
    
    local_data.fitness = local_best_fitness;
    local_data.rank = rank;
    
    MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    
    if (rank == global_data.rank) {
        for (int j = 0; j < DIM; j++) {
            global_best[j] = local_best[j];
        }
    }
    
    MPI_Bcast(global_best, DIM, MPI_DOUBLE, global_data.rank, MPI_COMM_WORLD);
    global_best_fitness = global_data.fitness;
}

void gather_all_solutions(FoodSource **all_foods) {
    int *recvcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    
    MPI_Allgather(&local_food_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + recvcounts[i-1];
    }
    
    int total_count = displs[size-1] + recvcounts[size-1];
    *all_foods = (FoodSource*)malloc(total_count * sizeof(FoodSource));
    
    MPI_Allgatherv(local_foods, local_food_count * sizeof(FoodSource), MPI_BYTE,
                   *all_foods, recvcounts, displs, MPI_BYTE, MPI_COMM_WORLD);
    
    free(recvcounts);
    free(displs);
}

void employed_bee_phase() {
    FoodSource *all_foods;
    gather_all_solutions(&all_foods);
    int total_foods = FOOD_NUMBER;
    
    for (int i = 0; i < local_food_count; i++) {
        double new_solution[DIM];
        
        for (int j = 0; j < DIM; j++) {
            new_solution[j] = local_foods[i].position[j];
        }
        
        int param = rand() % DIM;
        
        int neighbor = rand() % total_foods;
        
        double phi = random_double(-1.0, 1.0);
        new_solution[param] = local_foods[i].position[param] + 
                             phi * (local_foods[i].position[param] - all_foods[neighbor].position[param]);
        
        if (new_solution[param] < LOWER_BOUND) new_solution[param] = LOWER_BOUND;
        if (new_solution[param] > UPPER_BOUND) new_solution[param] = UPPER_BOUND;
        
        double new_fitness = calculate_fitness(new_solution);
        
        if (new_fitness > local_foods[i].fitness) {
            for (int j = 0; j < DIM; j++) {
                local_foods[i].position[j] = new_solution[j];
            }
            local_foods[i].fitness = new_fitness;
            local_foods[i].trial = 0;
        } else {
            local_foods[i].trial++;
        }
    }
    
    free(all_foods);
}

void onlooker_bee_phase() {
    FoodSource *all_foods;
    gather_all_solutions(&all_foods);
    int total_foods = FOOD_NUMBER;
    
    double total_fitness = 0.0;
    for (int i = 0; i < total_foods; i++) {
        total_fitness += all_foods[i].fitness;
    }
    
    for (int i = 0; i < local_food_count; i++) {
        double prob = local_foods[i].fitness / total_fitness;
        
        if (random_double(0.0, 1.0) < prob) {
            double new_solution[DIM];
            
            for (int j = 0; j < DIM; j++) {
                new_solution[j] = local_foods[i].position[j];
            }
            
            int param = rand() % DIM;
            
            int neighbor = rand() % total_foods;
            
            double phi = random_double(-1.0, 1.0);
            new_solution[param] = local_foods[i].position[param] + 
                                 phi * (local_foods[i].position[param] - all_foods[neighbor].position[param]);
            
            if (new_solution[param] < LOWER_BOUND) new_solution[param] = LOWER_BOUND;
            if (new_solution[param] > UPPER_BOUND) new_solution[param] = UPPER_BOUND;
            
            double new_fitness = calculate_fitness(new_solution);
            
            if (new_fitness > local_foods[i].fitness) {
                for (int j = 0; j < DIM; j++) {
                    local_foods[i].position[j] = new_solution[j];
                }
                local_foods[i].fitness = new_fitness;
                local_foods[i].trial = 0;
            } else {
                local_foods[i].trial++;
            }
        }
    }
    
    free(all_foods);
}

void scout_bee_phase() {
    for (int i = 0; i < local_food_count; i++) {
        if (local_foods[i].trial >= LIMIT) {
            for (int j = 0; j < DIM; j++) {
                local_foods[i].position[j] = random_double(LOWER_BOUND, UPPER_BOUND);
            }
            local_foods[i].fitness = calculate_fitness(local_foods[i].position);
            local_foods[i].trial = 0;
        }
    }
}

void print_results(int iteration) {
    if (rank == 0) {
        printf("Iteration %d: Best Fitness = %.10e\n", iteration, global_best_fitness);
    }
}

void print_usage(char *prog_name) {
    if (rank == 0) {
        printf("Usage: mpirun -np <processes> %s [OPTIONS]\n", prog_name);
        printf("\nOptions:\n");
        printf("  -d, --dim DIM              Problem dimension (default: 1500)\n");
        printf("  -f, --food FOOD            Number of food sources (default: 3000)\n");
        printf("  -i, --iter ITER            Maximum iterations (default: 40000)\n");
        printf("  -l, --log FILE             Convergence log file (default: convergence_mpi.csv)\n");
        printf("  -h, --help                 Show this help message\n");
        printf("\nExample:\n");
        printf("  mpirun -np 4 %s -d 1500 -f 3000 -i 50000 -l results.csv\n", prog_name);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char log_filename[256] = "convergence_mpi.csv";
    
    
    srand(time(NULL) + rank * 1000);
    
    local_food_count = FOOD_NUMBER / size;
    if (rank < FOOD_NUMBER % size) {
        local_food_count++;
    }
    
    local_foods = (FoodSource*)malloc(local_food_count * sizeof(FoodSource));
    
    if (rank == 0) {
        printf("MPI Artificial Bee Colony Algorithm\n");
        printf("====================================\n");
        printf("Problem Dimension: %d\n", DIM);
        printf("Total Food Sources: %d\n", FOOD_NUMBER);
        printf("Max Iterations: %d\n", MAX_ITER);
        printf("MPI Processes: %d\n", size);
        printf("Convergence Log: %s\n", log_filename);
        printf("Food sources per process: ");
        for (int i = 0; i < size; i++) {
            int count = FOOD_NUMBER / size;
            if (i < FOOD_NUMBER % size) count++;
            printf("%d ", count);
        }
        printf("\n\n");
        
        convergence_log = fopen(log_filename, "w");
        if (convergence_log == NULL) {
            printf("Warning: Could not open log file %s\n", log_filename);
        } else {
            fprintf(convergence_log, "Iteration,Time,Fitness\n");
        }
    }
    
    double start_time = MPI_Wtime();
    
    initialize_population();
    
    if (rank == 0) {
        printf("Initial Best Fitness: %.10e\n\n", global_best_fitness);
    }
    
    log_convergence(0, 0.0);
    
    for (int iter = 1; iter <= MAX_ITER; iter++) {
        if (global_best_fitness < 1e-2) { 
            break;
        }
        employed_bee_phase();
        onlooker_bee_phase();
        scout_bee_phase();
        
        communicate_best_solution();
        
        double elapsed = MPI_Wtime() - start_time;
        
        if (iter % 10 == 0) {
            log_convergence(iter, elapsed);
        }
        
        if (iter % 100 == 0) {
            print_results(iter);
        }
    }
    
    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;
    
    log_convergence(MAX_ITER, total_time);
    
    if (rank == 0) {
        printf("\n====================================\n");
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
    }
    
    free(local_foods);
    
    MPI_Finalize();
    return 0;
}