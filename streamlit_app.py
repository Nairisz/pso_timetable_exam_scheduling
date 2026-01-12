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
iterations = st.sidebar.slider("Iterations", 50, 350, 200, 50)
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

    total_seconds = int(result["runtime"])
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    formatted_runtime = f"{minutes} min {seconds:02d} sec"

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Fitness", f"{result['fitness']:.2f}")
    col2.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
    col3.metric("Runtime", formatted_runtime)

    # -------- Convergence --------
    # -------- Styled Convergence Curve --------
    st.subheader("📉 Convergence Curve")

    fig, ax = plt.subplots(figsize=(10, 4))

    # Dark background
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Plot smooth line
    ax.plot(
        result["convergence"],
        linewidth=2.5,
        color="#4da3ff"
    )

    # Subtle grid
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.3
    )

    # Axis styling
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Iterations", color="white")
    ax.set_ylabel("Best Fitness", color="white")

    st.pyplot(fig)
    
    # -------- Text-style Timetable (Per Day) --------
    st.subheader("🗓️ Final Exam Schedule")

    solution = result["solution"]

    timeslot_labels = [
        "09:00 AM", "10:00 AM", "11:00 AM",
        "12:00 PM", "01:00 PM", "02:00 PM",
        "03:00 PM", "04:00 PM"
    ][:num_timeslots]

    days = exams["exam_day"].unique()

    for day in days:
        with st.expander(f"Exam Schedule for Day {day} ", expanded=True):

            schedule_map = set()
            schedule = {t: [] for t in timeslot_labels}

            exams_day = exams[exams["exam_day"] == day]

            for _, row in exams_day.iterrows():
                idx = exams[exams["exam_id"] == row["exam_id"]].index[0]

                ts = int(np.clip(round(solution[2*idx]), 0, num_timeslots - 1))
                rm = int(np.clip(round(solution[2*idx + 1]), 0, len(rooms) - 1))

                ts, rm = repair_solution(
                    ts, rm, idx, exams, rooms, num_timeslots, schedule_map
                )

                schedule_map.add((ts, rm))

                room = rooms.iloc[rm]
                entry = (
                    f"{row['course_code']} ({row['exam_type']}) - "
                    f"{room['building_name']} {room['room_number']} "
                    f"[{room['room_type']}]"
                )

                schedule[timeslot_labels[ts]].append(entry)

            # ✅ Render INSIDE expander
            for time in timeslot_labels:
                st.markdown(f"**Time Slot {time}**")
                if not schedule[time]:
                    st.write("  - Free")
                else:
                    for exam in schedule[time]:
                        st.write(f"  - {exam}")
