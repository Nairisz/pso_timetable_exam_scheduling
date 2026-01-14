import numpy as np
import random
import time

# =========================
# Repair Function (ROOM ONLY)
# =========================
def repair_solution(room, exam_idx, exams, rooms, schedule_map):
    students = exams.iloc[exam_idx]["num_students"]
    exam_type = exams.iloc[exam_idx]["exam_type"].lower()
    day = exams.iloc[exam_idx]["exam_day"]
    time_slot = exams.iloc[exam_idx]["exam_time"]

    num_rooms = len(rooms)

    # ---- Capacity repair ----
    if students > rooms.iloc[room]["capacity"]:
        feasible = rooms[rooms["capacity"] >= students].index.tolist()
        if feasible:
            room = random.choice(feasible)

    # ---- Room type repair ----
    room_type = rooms.iloc[room]["room_type"].lower()

    if exam_type == "practical" and "lab" not in room_type:
        labs = rooms[rooms["room_type"].str.lower().str.contains("lab")].index.tolist()
        if labs:
            room = random.choice(labs)

    if exam_type == "theory" and "lab" in room_type:
        non_labs = rooms[~rooms["room_type"].str.lower().str.contains("lab")].index.tolist()
        if non_labs:
            room = random.choice(non_labs)

    # ---- Clash repair (same day, same time, same room) ----
    key = (day, time_slot, room)
    if key in schedule_map:
        free_rooms = [
            r for r in range(num_rooms)
            if (day, time_slot, r) not in schedule_map
        ]
        if free_rooms:
            room = random.choice(free_rooms)

    return room


# =========================
# Fitness Function
# =========================
def fitness_multiobj(solution, exams, rooms):
    num_exams = len(exams)
    num_rooms = len(rooms)

    penalty_constraints = 0
    room_usage = np.zeros(num_rooms)
    schedule_map = set()  # (day, time, room)

    for i in range(num_exams):
        room = int(np.clip(round(solution[i]), 0, num_rooms - 1))

        students = exams.iloc[i]["num_students"]
        exam_type = exams.iloc[i]["exam_type"].lower()
        day = exams.iloc[i]["exam_day"]
        time_slot = exams.iloc[i]["exam_time"]

        room_info = rooms.iloc[room]
        capacity = room_info["capacity"]
        room_type = room_info["room_type"].lower()

        needs_repair = (
            students > capacity or
            (exam_type == "practical" and "lab" not in room_type) or
            (exam_type == "theory" and "lab" in room_type) or
            ((day, time_slot, room) in schedule_map)
        )

        if needs_repair:
            room = repair_solution(room, i, exams, rooms, schedule_map)

        # ---- Hard constraint penalties ----
        if students > rooms.iloc[room]["capacity"]:
            penalty_constraints += 5

        if exam_type == "practical" and "lab" not in rooms.iloc[room]["room_type"].lower():
            penalty_constraints += 3

        if exam_type == "theory" and "lab" in rooms.iloc[room]["room_type"].lower():
            penalty_constraints += 2

        key = (day, time_slot, room)
        if key in schedule_map:
            penalty_constraints += 10
        else:
            schedule_map.add(key)

        room_usage[room] += students

    # ---- Soft objective: utilization variance ----
    penalty_util = (
        np.var(room_usage / np.sum(room_usage))
        if np.sum(room_usage) > 0 else 0
    )

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
    w,
    c1,
    c2,
    progress_callback=None
):
    np.random.seed(42)
    random.seed(42)
    start_time = time.time()

    num_exams = len(exams)
    num_rooms = len(rooms)

    # ---- ROOM-ONLY encoding ----
    particles = np.random.rand(num_particles, num_exams) * num_rooms
    velocities = np.zeros_like(particles)

    pbest = particles.copy()
    pbest_fit = np.array([
        fitness_multiobj(p, exams, rooms)[0]
        for p in particles
    ])

    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    convergence = []

    # ===== PSO main loop =====
    for it in range(iterations):
        for i in range(num_particles):
            r1, r2 = random.random(), random.random()

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )

            particles[i] += velocities[i]

            fit, _ = fitness_multiobj(particles[i], exams, rooms)

            if fit < pbest_fit[i]:
                pbest[i] = particles[i].copy()
                pbest_fit[i] = fit

                if fit < gbest_fit:
                    gbest = particles[i].copy()
                    gbest_fit = fit

        convergence.append(gbest_fit)

        # ✅ Progress update
        if progress_callback:
            progress_callback((it + 1) / iterations)

    runtime = time.time() - start_time

    # ===== Accuracy Calculation =====
    _, total_violations = fitness_multiobj(gbest, exams, rooms)

    constraint_accuracy = max(
        0, 1 - total_violations / (num_exams * 10)
    )

    room_usage = np.zeros(num_rooms)
    for i in range(num_exams):
        room = int(np.clip(round(gbest[i]), 0, num_rooms - 1))
        room_usage[room] += exams.iloc[i]["num_students"]

    utilization_accuracy = (
        1 / (1 + np.var(room_usage / np.sum(room_usage)))
        if np.sum(room_usage) > 0 else 1
    )

    final_accuracy = 0.7 * constraint_accuracy + 0.3 * utilization_accuracy

    return {
        "solution": gbest,
        "fitness": gbest_fit,
        "constraint_accuracy": constraint_accuracy,
        "utilization_accuracy": utilization_accuracy,
        "accuracy": final_accuracy,
        "runtime": runtime,
        "convergence": convergence
    }
