import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pso_core import run_pso

# ================= Page setup =================
st.set_page_config(layout="wide")

with open("theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📘 PSO Exam Timetabling Dashboard")
st.markdown("---")

# ================= Explanation =================
with st.expander("ℹ️ About the Optimization (How PSO Works)", expanded=False):
    st.markdown("""
**Particle Swarm Optimization (PSO)** is a population-based optimization algorithm inspired by swarm intelligence.

In this system:
- Each **particle represents a room assignment for all exams**
- Exam **days and times are fixed** by the dataset
- PSO optimizes **room allocation only**

Particles update their positions using:
- **Personal best solution** (cognitive component)
- **Global best solution** (social component)
""")

with st.expander("📌 Optimization Objectives & Rules", expanded=False):
    st.markdown("""
### 🎯 Optimization Goal
Minimize total scheduling cost while respecting hard constraints.

### ❌ Hard Constraints
1. Room capacity must be sufficient  
2. Practical exams must use laboratories  
3. Theory exams must not occupy labs  
4. No room conflict at the same day & time  

### ⚖️ Soft Objective
- Reduce **wasted seating capacity**

**Fitness = Hard Constraint Penalties + Room Utilization Variance**
""")

# ================= Load data =================
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

    with col1:
        st.markdown("### 📘 Exams")
        st.write(f"**Total Exams:** {len(exams)}")
        st.write(f"**Exam Days:** {exams['exam_day'].nunique()}")
        for k, v in exams["exam_type"].value_counts().items():
            st.write(f"• {k}: {v}")

        st.markdown("**Students per Exam**")
        st.write(f"Min: {exams['num_students'].min()}")
        st.write(f"Max: {exams['num_students'].max()}")
        st.write(f"Average: {exams['num_students'].mean():.1f}")

    with col2:
        st.markdown("### 🏫 Rooms")
        st.write(f"**Total Rooms:** {len(rooms)}")
        for k, v in rooms["room_type"].value_counts().items():
            st.write(f"• {k}: {v}")

        st.write(f"**Total Seats Available:** {rooms['capacity'].sum()}")

    st.markdown("---")
    if st.checkbox("Show exams table"):
        st.dataframe(exams, use_container_width=True)
    if st.checkbox("Show rooms table"):
        st.dataframe(rooms, use_container_width=True)

# ================= Sidebar =================
st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.slider("Particles", 10, 200, 50, 10)
iterations = st.sidebar.slider("Iterations", 50, 300, 50, 50)

w = st.sidebar.slider("Inertia Weight (w)", 0.1, 1.2, 0.7, 0.1)
c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.5, 3.0, 1.5, 0.1)
c2 = st.sidebar.slider("Social Coefficient (c2)", 0.5, 3.0, 1.5, 0.1)

run = st.sidebar.button("🚀 Run PSO")

# ================= Run PSO =================
progress_bar = st.progress(0)
status_text = st.empty()

    def update_progress(p):
        progress_bar.progress(p)
        status_text.text(f"Running PSO... {int(p * 100)}%")

    result = run_pso(
        exams,
        rooms,
        num_particles,
        iterations,
        num_timeslots,
        w,
        c1,
        c2,
        progress_callback=update_progress
    )

    progress_bar.empty()
    status_text.empty()

    solution = result["solution"
    st.success("Optimization completed")

    # ================= Metrics =================
    curve = np.array(result["convergence"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Fitness", f"{result['fitness']:.2f}")
    col2.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
    col3.metric("Runtime", f"{int(result['runtime']//60)}m {int(result['runtime']%60)}s")
    col4.metric("Iterations", len(curve))

    # ================= Convergence =================
    st.subheader("📉 Convergence Curve")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(curve, color="#80A1BA", linewidth=2.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

    # ================= Capacity metrics =================
    solution = result["solution"]

    capacity_violations = 0
    wasted_seats = 0
    used_seats = 0

    for i in range(len(exams)):
        room = int(np.clip(round(solution[i]), 0, len(rooms) - 1))
        students = exams.iloc[i]["num_students"]
        capacity = rooms.iloc[room]["capacity"]

        used_seats += students
        if students > capacity:
            capacity_violations += 1
        else:
            wasted_seats += (capacity - students)

    col5, col6, col7 = st.columns(3)
    col5.metric("Capacity Violations", capacity_violations)
    col6.metric("Wasted Seats", wasted_seats)
    col7.metric("Wasted Capacity Ratio", f"{wasted_seats/(used_seats+wasted_seats)*100:.2f}%")

   # ================= Final Timetable =================
st.subheader("🗓️ Final Exam Schedule")

for day in exams["exam_day"].unique():
    with st.expander(f"Day {day}", expanded=True):

        exams_day = exams[exams["exam_day"] == day]

        for _, row in exams_day.iterrows():
            idx = exams.index[exams["exam_id"] == row["exam_id"]][0]

            # PSO decides ROOM only (timeslot is fixed from dataset)
            room_idx = int(np.clip(round(solution[idx]), 0, len(rooms) - 1))

            exam_type = row["exam_type"].lower()
            room_type = rooms.iloc[room_idx]["room_type"].lower()

            # ===== HARD ENFORCEMENT =====
            # Practical → Lab
            if exam_type == "practical" and "lab" not in room_type:
                lab_rooms = rooms[
                    rooms["room_type"].str.lower().str.contains("lab")
                ].index.tolist()
                if lab_rooms:
                    room_idx = random.choice(lab_rooms)

            # Theory → Non-lab
            if exam_type == "theory" and "lab" in room_type:
                non_lab_rooms = rooms[
                    ~rooms["room_type"].str.lower().str.contains("lab")
                ].index.tolist()
                if non_lab_rooms:
                    room_idx = random.choice(non_lab_rooms)

            room = rooms.iloc[room_idx]

            # ===== DISPLAY =====
            st.write(
                f"**{row['exam_id']} | {row['course_code']} | {row['exam_type']}**  \n"
                f"Room: {room['building_name']} {room['room_number']} "
                f"({room['room_type']})  \n"
                f"Time: {row['exam_time']}"
            )
