import numpy as np
import random
import time

def run_pso(
    exams,
    rooms,
    num_timeslots,
    num_particles,
    iterations,
    w,
    c1,
    c2,
    fitness_fn
):
    num_exams = len(exams)
    num_rooms = len(rooms)
    dimensions = num_exams * 2

    particles = np.random.rand(num_particles, dimensions)
    velocities = np.zeros_like(particles)

    for p in range(num_particles):
        for i in range(num_exams):
            particles[p][2*i] *= num_timeslots
            particles[p][2*i + 1] *= num_rooms

    personal_best = particles.copy()
    personal_best_fitness = np.array([fitness_fn(p)[0] for p in particles])

    gbest_idx = np.argmin(personal_best_fitness)
    global_best = personal_best[gbest_idx].copy()
    global_best_fitness = personal_best_fitness[gbest_idx]

    convergence = []
    start = time.time()

    for _ in range(iterations):
        for i in range(num_particles):
            r1, r2 = random.random(), random.random()

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - particles[i])
                + c2 * r2 * (global_best - particles[i])
            )
            particles[i] += velocities[i]

            fit, _ = fitness_fn(particles[i])
            if fit < personal_best_fitness[i]:
                personal_best[i] = particles[i].copy()
                personal_best_fitness[i] = fit

                if fit < global_best_fitness:
                    global_best = particles[i].copy()
                    global_best_fitness = fit

        convergence.append(global_best_fitness)

    runtime = time.time() - start
    _, violations = fitness_fn(global_best)

    accuracy = max(0, 1 - violations / (num_exams * 10))

    return {
        "best_fitness": global_best_fitness,
        "violations": violations,
        "accuracy": accuracy,
        "runtime": runtime,
        "convergence": convergence,
        "solution": global_best
    }
