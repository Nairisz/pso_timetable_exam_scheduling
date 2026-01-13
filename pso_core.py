import numpy as np
import random
import time

# =========================
# Repair Function
# =========================
def repair_solution(timeslot, room, exam_idx, exams, rooms, num_timeslots, schedule_map):
    students = exams.iloc[exam_idx]["num_students"]
    exam_type = exams.iloc[exam_idx]["exam_type"].lower()
    num_rooms = len(rooms)

    if students > rooms.iloc[room]["capacity"]:
        feasible = rooms[rooms["capacity"] >= students].index.tolist()
        if feasible:
            room = random.choice(feasible)

    room_type = rooms.iloc[room]["room_type"].lower()
    if exam_type == "practical" and "lab" not in room_type:
        labs = rooms[rooms["room_type"].str.lower().str.contains("lab")].index.tolist()
        if labs:
            room = random.choice(labs)

    if exam_type == "theory" and "lab" in room_type:
        non_labs = rooms[~rooms["room_type"].str.lower().str.contains("lab")].index.tolist()
        if non_labs:
            room = random.choice(non_labs)

    if (timeslot, room) in schedule_map:
        free_rooms = [r for r in range(num_rooms) if (timeslot, r) not in schedule_map]
        if free_rooms:
            room = random.choice(free_rooms)
        else:
            timeslot = (timeslot + 1) % num_timeslots

    return timeslot, room


# =========================
# Fitness Function
# =========================
def fitness_multiobj(solution, exams, rooms, num_timeslots):
    num_exams = len(exams)
    num_rooms = len(rooms)

    penalty_constraints = 0
    room_usage = np.zeros(num_rooms)
    schedule_map = set()

    for i in range(num_exams):
        timeslot = int(np.clip(round(solution[2*i]), 0, num_timeslots - 1))
        room = int(np.clip(round(solution[2*i + 1]), 0, num_rooms - 1))

        students = exams.iloc[i]["num_students"]
        exam_type = exams.iloc[i]["exam_type"].lower()
        room_type = rooms.iloc[room]["room_type"].lower()
        capacity = rooms.iloc[room]["capacity"]

        needs_repair = (
            students > capacity or
            (exam_type == "practical" and "lab" not in room_type) or
            (exam_type == "theory" and "lab" in room_type) or
            ((timeslot, room) in schedule_map)
        )

        if needs_repair:
            timeslot, room = repair_solution(
                timeslot, room, i, exams, rooms, num_timeslots, schedule_map
            )

        # penalties
        if students > rooms.iloc[room]["capacity"]:
            penalty_constraints += 5

        if exam_type == "practical" and "lab" not in rooms.iloc[room]["room_type"].lower():
            penalty_constraints += 3

        if exam_type == "theory" and "lab" in rooms.iloc[room]["room_type"].lower():
            penalty_constraints += 2

        if (timeslot, room) in schedule_map:
            penalty_constraints += 10

        schedule_map.add((timeslot, room))
        room_usage[room] += students

    penalty_util = np.var(room_usage / np.sum(room_usage)) if np.sum(room_usage) > 0 else 0
    total_fitness = penalty_constraints + penalty_util


    return total_fitness, penalty_constraints


# =========================
# PSO Runner
# =========================
def run_pso(
    exams,
    rooms,
    num_particles,
    iterations,
    num_timeslots,
    w,
    c1,
    c2
):
    np.random.seed(42)
    random.seed(42)
    start_time = time.time()

    num_exams = len(exams)
    num_rooms = len(rooms)
    dimensions = num_exams * 2

    particles = np.random.rand(num_particles, dimensions)
    velocities = np.zeros_like(particles)

    # Scale particles
    for p in range(num_particles):
        for i in range(num_exams):
            particles[p][2*i] *= num_timeslots
            particles[p][2*i + 1] *= num_rooms

    pbest = particles.copy()
    pbest_fit = np.array([
        fitness_multiobj(p, exams, rooms, num_timeslots)[0]
        for p in particles
    ])

    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    convergence = []

    # ===== PSO loop =====
    for _ in range(iterations):
        for i in range(num_particles):
            r1, r2 = random.random(), random.random()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )
            particles[i] += velocities[i]

            fit, _ = fitness_multiobj(particles[i], exams, rooms, num_timeslots)

            if fit < pbest_fit[i]:
                pbest[i] = particles[i].copy()
                pbest_fit[i] = fit

                if fit < gbest_fit:
                    gbest = particles[i].copy()
                    gbest_fit = fit

        convergence.append(gbest_fit)

    runtime = time.time() - start_time

    # ===== Accuracy calculation (INSIDE run_pso) =====
    _, total_violations = fitness_multiobj(gbest, exams, rooms, num_timeslots)

    constraint_accuracy = 1 - (total_violations / (num_exams * 10))
    constraint_accuracy = max(0, constraint_accuracy)

    room_usage = np.zeros(num_rooms)
    for i in range(num_exams):
        room = int(np.clip(round(gbest[2*i + 1]), 0, num_rooms - 1))
        room_usage[room] += exams.iloc[i]["num_students"]

    if np.sum(room_usage) > 0:
        utilization_penalty = np.var(room_usage / np.sum(room_usage))
        utilization_accuracy = 1 / (1 + utilization_penalty)
    else:
        utilization_accuracy = 1

    final_accuracy = (
        0.7 * constraint_accuracy +
        0.3 * utilization_accuracy
    )

    return {
        "solution": gbest,
        "fitness": gbest_fit,
        "constraint_accuracy": constraint_accuracy,
        "utilization_accuracy": utilization_accuracy,
        "accuracy": final_accuracy,
        "runtime": runtime,
        "convergence": convergence
    }
