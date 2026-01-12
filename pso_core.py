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

    schedule_map.add((timeslot, room))
    return timeslot, room


# =========================
# Fitness Function
# =========================
def fitness_multiobj(solution, exams, rooms, num_timeslots):
    num_exams = len(exams)
    num_rooms = len(rooms)

    penalty = 0
    timeslot_set = set()
    room_usage = np.zeros(num_rooms)
    schedule_map = set()

    for i in range(num_exams):
        timeslot = int(np.clip(round(solution[2*i]), 0, num_timeslots-1))
        room = int(np.clip(round(solution[2*i+1]), 0, num_rooms-1))

        timeslot, room = repair_solution(
            timeslot, room, i, exams, rooms, num_timeslots, schedule_map
        )

        students = exams.iloc[i]["num_students"]
        capacity = rooms.iloc[room]["capacity"]
        exam_type = exams.iloc[i]["exam_type"].lower()
        room_type = rooms.iloc[room]["room_type"].lower()

        if students > capacity:
            penalty += 5
        if exam_type == "practical" and "lab" not in room_type:
            penalty += 3
        if exam_type == "theory" and "lab" in room_type:
            penalty += 2

        for j in range(i + 1, num_exams):
            other_ts = int(np.clip(round(solution[2*j]), 0, num_timeslots-1))
            other_room = int(np.clip(round(solution[2*j+1]), 0, num_rooms-1))
            if timeslot == other_ts and room == other_room:
                penalty += 10
            elif timeslot == other_ts:
                penalty += 2

        timeslot_set.add(timeslot)
        room_usage[room] += students

    penalty_timeslot = 3 * len(timeslot_set)
    penalty_util = np.var(room_usage / np.sum(room_usage)) if np.sum(room_usage) > 0 else 0

    total = penalty + penalty_timeslot + penalty_util
    return total, penalty


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
    start_time = time.time()

    num_exams = len(exams)
    num_rooms = len(rooms)
    dimensions = num_exams * 2

    particles = np.random.rand(num_particles, dimensions)
    velocities = np.zeros_like(particles)

    for p in range(num_particles):
        for i in range(num_exams):
            particles[p][2*i] *= num_timeslots
            particles[p][2*i+1] *= num_rooms

    pbest = particles.copy()
    pbest_fit = np.array([
        fitness_multiobj(p, exams, rooms, num_timeslots)[0]
        for p in particles
    ])

    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    convergence = []

    for _ in range(iterations):
        for i in range(num_particles):
            r1, r2 = random.random(), random.random()

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )
            particles[i] += velocities[i]

            fit, _ = fitness_multiobj(
                particles[i], exams, rooms, num_timeslots
            )

            if fit < pbest_fit[i]:
                pbest[i] = particles[i].copy()
                pbest_fit[i] = fit

                if fit < gbest_fit:
                    gbest = particles[i].copy()
                    gbest_fit = fit

        convergence.append(gbest_fit)

    runtime = time.time() - start_time
    _, violations = fitness_multiobj(gbest, exams, rooms, num_timeslots)
    accuracy = max(0, 1 - violations / (num_exams * 10))

    return {
        "best_solution": gbest,
        "best_fitness": gbest_fit,
        "convergence": convergence,
        "runtime": runtime,
        "accuracy": accuracy
    }
