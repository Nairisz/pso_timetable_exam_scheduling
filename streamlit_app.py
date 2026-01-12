import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pso_core import run_pso, repair_solution

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
    exams = pd.read_csv("exam_timeslot.csv")
    rooms = pd.read_csv("classrooms.csv")
    exams.columns = exams.columns.str.strip().str.lower()
    rooms.columns = rooms.columns.str.strip().str.lower()
    return exams, rooms

exams, rooms = load_data()

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.slider("Particles", 10, 200, 50, 10)
iterations = st.sidebar.slider("Iterations", 50, 1000, 300, 50)
num_timeslots = st.sidebar.slider("Timeslots", 3, 10, 5)

w = st.sidebar.slider("w", 0.1, 1.2, 0.9, 0.1)
c1 = st.sidebar.slider("c1", 0.5, 3.0, 2.0, 0.1)
c2 = st.sidebar.slider("c2", 0.5, 3.0, 1.2, 0.1)

run = st.sidebar.button("🚀 Run PSO")

# =========================
# Run PSO
# =========================
if run:
    with st.spinner("Optimizing..."):
        results = run_pso(
            exams,
            rooms,
            num_particles,
            iterations,
            num_timeslots,
            w,
            c1,
            c2
        )

    st.success("Done ✨")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Fitness", f"{results['best_fitness']:.2f}")
    col2.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
    col3.metric("Runtime (s)", f"{results['runtime']:.2f}")

    # =========================
    # Convergence Plot
    # =========================
    st.subheader("📉 Convergence Curve")
    fig = plt.figure()
    plt.plot(results["convergence"])
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    st.pyplot(fig)

    # =========================
    # Timetable
    # =========================
    st.subheader("🗓️ Final Exam Timetable")

    timeslot_labels = [
        "09:00 AM", "10:00 AM", "11:00 AM",
        "12:00 PM", "01:00 PM", "02:00 PM",
        "03:00 PM", "04:00 PM"
    ][:num_timeslots]

    solution = results["best_solution"]
    schedule_map = set()

    for i in range(len(exams)):
        ts = int(np.clip(round(solution[2*i]), 0, num_timeslots-1))
        rm = int(np.clip(round(solution[2*i+1]), 0, len(rooms)-1))

        ts, rm = repair_solution(
            ts, rm, i, exams, rooms, num_timeslots, schedule_map
        )

        room = rooms.iloc[rm]
        st.write(
            f"{timeslot_labels[ts]} → "
            f"{exams.iloc[i]['course_code']} "
            f"({exams.iloc[i]['exam_type']}) | "
            f"{room['building_name']} {room['room_number']}"
        )
