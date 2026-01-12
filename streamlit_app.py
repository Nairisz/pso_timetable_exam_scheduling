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

# ================= Dataset Overview =================
st.subheader("📊 Dataset Overview")

with st.expander("View dataset summary", expanded=True):

    col1, col2 = st.columns(2)

    # ---- Exams Overview ----
    with col1:
        st.markdown("### 📘 Exams Dataset")

        st.write(f"**Total Exams:** {len(exams)}")
        st.write(f"**Exam Days:** {exams['exam_day'].nunique()}")

        exam_type_counts = exams['exam_type'].value_counts()
        for etype, count in exam_type_counts.items():
            st.write(f"• {etype}: {count}")

        st.markdown("**Students per Exam**")
        st.write(f"Min: {exams['num_students'].min()}")
        st.write(f"Max: {exams['num_students'].max()}")
        st.write(f"Average: {exams['num_students'].mean():.1f}")
         
    # -------- Raw Tables (BOTTOM of expander) --------
    st.markdown("---")  # visual separator
    st.markdown("### 📄 Raw Dataset Tables")

    show_exams = st.checkbox("Show exams table")
    if show_exams:
        st.dataframe(exams, use_container_width=True)

    show_rooms = st.checkbox("Show rooms table")
    if show_rooms:
        st.dataframe(rooms, use_container_width=True)

    # ---- Rooms Overview ----
    with col2:
        st.markdown("### 🏫 Rooms Dataset")

        st.write(f"**Total Rooms:** {len(rooms)}")

        room_type_counts = rooms['room_type'].value_counts()
        for rtype, count in room_type_counts.items():
            st.write(f"• {rtype}: {count}")

        st.markdown("**Room Capacity**")
        st.write(f"Min: {rooms['capacity'].min()}")
        st.write(f"Max: {rooms['capacity'].max()}")
        st.write(f"Average: {rooms['capacity'].mean():.1f}")

# ================= Sidebar =================
st.sidebar.header("⚙️ Parameters")
num_particles = st.sidebar.slider("Particles", 10, 200, 50, 10)
iterations = st.sidebar.slider("Iterations", 50, 350, 200, 50)
num_timeslots = st.sidebar.slider("Timeslots", 3, 8, 5)
w = st.sidebar.slider(
    "Inertia Weight (w)",
    min_value=0.1,
    max_value=1.2,
    value=0.9,
    step=0.1,
    help="Controls momentum of particles. Higher = more exploration."
)

c1 = st.sidebar.slider(
    "Cognitive Coefficient (c1)",
    min_value=0.5,
    max_value=3.0,
    value=2.0,
    step=0.1,
    help="Influence of particle's personal best."
)

c2 = st.sidebar.slider(
    "Social Coefficient (c2)",
    min_value=0.5,
    max_value=3.0,
    value=1.2,
    step=0.1,
    help="Influence of global best solution."
)

st.sidebar.caption(
    "Tip: Lower inertia = faster convergence, "
    "higher inertia = wider exploration."
)

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
