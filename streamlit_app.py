import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pso_core import run_pso, repair_solution

st.set_page_config(layout="wide")
st.title("📘 PSO Exam Timetabling Dashboard")

@st.cache_data
def load_data():
    exams = pd.read_csv("exam_timeslot.csv")
    rooms = pd.read_csv("classrooms.csv")
    exams.columns = exams.columns.str.strip().str.lower()
    rooms.columns = rooms.columns.str.strip().str.lower()
    return exams, rooms

exams, rooms = load_data()

# ================= Sidebar =================
st.sidebar.header("⚙️ Parameters")
num_particles = st.sidebar.slider("Particles", 10, 200, 50, 10)
iterations = st.sidebar.slider("Iterations", 50, 350, 300, 50)
num_timeslots = st.sidebar.slider("Timeslots", 3, 8, 5)
w = st.sidebar.slider("w", 0.1, 1.2, 0.9, 0.1)
c1 = st.sidebar.slider("c1", 0.5, 3.0, 2.0, 0.1)
c2 = st.sidebar.slider("c2", 0.5, 3.0, 1.2, 0.1)

run = st.sidebar.button("🚀 Run PSO")

# ================= Run =================
if run:
    with st.spinner("Optimizing timetable..."):
        result = run_pso(
            exams,
            rooms,
            num_particles,
            iterations,
            num_timeslots,
            w,
            c1,
            c2
        )

    st.success("Optimization completed")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Fitness", f"{result['fitness']:.2f}")
    col2.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
    col3.metric("Runtime (s)", f"{result['runtime']:.2f}")

    # -------- Convergence --------
    st.subheader("📉 Convergence Curve")
    fig = plt.figure()
    plt.plot(result["convergence"])
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    st.pyplot(fig)

    # -------- Timetable --------
    st.subheader("🗓️ Final Exam Timetable")

    labels = [
        "09:00 AM", "10:00 AM", "11:00 AM",
        "12:00 PM", "01:00 PM", "02:00 PM",
        "03:00 PM", "04:00 PM"
    ][:num_timeslots]

    solution = result["solution"]
    schedule_map = set()

    for i in range(len(exams)):
        ts = int(np.clip(round(solution[2*i]), 0, num_timeslots - 1))
        rm = int(np.clip(round(solution[2*i + 1]), 0, len(rooms) - 1))

        ts, rm = repair_solution(ts, rm, i, exams, rooms, num_timeslots, schedule_map)
        schedule_map.add((ts, rm))

        room = rooms.iloc[rm]
        st.write(
            f"{labels[ts]} → "
            f"{exams.iloc[i]['course_code']} "
            f"({exams.iloc[i]['exam_type']}) | "
            f"{room['building_name']} {room['room_number']} [{room['room_type']}]"
        )
