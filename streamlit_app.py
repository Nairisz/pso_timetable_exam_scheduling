import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time

# =========================
# Page Config
# =========================
st.set_page_config(page_title="PSO Exam Timetabling", layout="wide")
st.title("📘 PSO-based Exam Timetabling System")

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    exams = ("exam_timeslot.csv")
    rooms = ("classrooms.csv")

    exams.columns = exams.columns.str.strip().str.lower()
    rooms.columns = rooms.columns.str.strip().str.lower()
    return exams, rooms

exams, rooms = load_data()

num_exams = len(exams)
num_rooms = len(rooms)

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.slider("Number of Particles", 10, 200, 50, step=10)
iterations = st.sidebar.slider("Iterations", 50, 1000, 300, step=50)
num_timeslots = st.sidebar.slider("Number of Timeslots", 3, 10, 5)

w = st.sidebar.slider("Inertia Weight (w)", 0.1, 1.2, 0.9, step=0.1)
c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.5, 3.0, 2.0, step=0.1)
c2 = st.sidebar.slider("Social Coefficient (c2)", 0.5, 3.0, 1.2, step=0.1)

run_button = st.sidebar.button("🚀 Run PSO")

# =========================
# Repair Function
# =========================
def repair_solution(timeslot, room, exam_idx, schedule_map):
    students = exams.iloc[exam_idx]["num_students"]
    exam_type = exams.iloc[exam_idx]["exam_type"].lower()

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
def fitness_multiobj(solution):
    penalty = 0
    timeslot_set = set()
    room_usage = np.zeros(num_rooms)
    schedule_map = set()

    for i in range(num_exams):
        timeslot = int(np.clip(round(solution[2*i]), 0, num_timeslots-1))
        room = int(np.clip(round(solution[2*i+1]), 0, num_rooms-1))

        timeslot, room = repair_solution(timeslot, room, i, schedule_map)

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
# Run PSO
# =========================
if run_button:
    with st.spinner("Running PSO optimization..."):
        start = time.time()

        dimensions = num_exams * 2
        particles = np.random.rand(num_particles, dimensions)
        velocities = np.zeros_like(particles)

        for p in range(num_particles):
            for i in range(num_exams):
                particles[p][2*i] *= num_timeslots
                particles[p][2*i+1] *= num_rooms

        pbest = particles.copy()
        pbest_fit = np.array([fitness_multiobj(p)[0] for p in particles])
        gbest = pbest[np.argmin(pbest_fit)].copy()
        gbest_fit = np.min(pbest_fit)

        convergence = []

        for it in range(iterations):
            for i in range(num_particles):
                r1, r2 = random.random(), random.random()
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (pbest[i] - particles[i])
                    + c2 * r2 * (gbest - particles[i])
                )
                particles[i] += velocities[i]

                fit, _ = fitness_multiobj(particles[i])
                if fit < pbest_fit[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fit[i] = fit
                    if fit < gbest_fit:
                        gbest = particles[i].copy()
                        gbest_fit = fit

            convergence.append(gbest_fit)

        runtime = time.time() - start
        _, violations = fitness_multiobj(gbest)
        accuracy = max(0, 1 - violations / (num_exams * 10))

    # =========================
    # Results
    # =========================
    st.success("Optimization completed")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Fitness", f"{gbest_fit:.2f}")
    col2.metric("Accuracy", f"{accuracy*100:.2f}%")
    col3.metric("Runtime (s)", f"{runtime:.2f}")

    # =========================
    # Convergence Plot
    # =========================
    st.subheader("📉 Convergence Curve")
    fig = plt.figure()
    plt.plot(convergence)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    st.pyplot(fig)

    # =========================
    # Final Timetable
    # =========================
    st.subheader("🗓️ Final Exam Timetable")

    timeslot_labels = [
        "09:00 AM", "10:00 AM", "11:00 AM",
        "12:00 PM", "01:00 PM", "02:00 PM",
        "03:00 PM", "04:00 PM"
    ][:num_timeslots]

    days = exams["exam_day"].unique()

    for day in days:
        st.markdown(f"### Day {day}")
        schedule_map = set()
        schedule = {t: [] for t in timeslot_labels}

        exams_day = exams[exams["exam_day"] == day]

        for _, row in exams_day.iterrows():
            idx = exams[exams["exam_id"] == row["exam_id"]].index[0]
            ts = int(np.clip(round(gbest[2*idx]), 0, num_timeslots-1))
            rm = int(np.clip(round(gbest[2*idx+1]), 0, num_rooms-1))
            ts, rm = repair_solution(ts, rm, idx, schedule_map)

            room = rooms.iloc[rm]
            schedule[timeslot_labels[ts]].append(
                f"{row['course_code']} ({row['exam_type']}) — "
                f"{room['building_name']} {room['room_number']} [{room['room_type']}]"
            )

        for t, items in schedule.items():
            st.markdown(f"**{t}**")
            if items:
                for item in items:
                    st.write("•", item)
            else:
                st.write("• Free")
