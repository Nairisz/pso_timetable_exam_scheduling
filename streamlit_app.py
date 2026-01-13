import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pso_core import run_pso, repair_solution

st.set_page_config(layout="wide")
with open("theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📘 PSO Exam Timetabling Dashboard")

st.markdown("---")

with st.expander("ℹ️ About the Optimization (How PSO Works)", expanded=False):
    st.markdown("""
**Particle Swarm Optimization (PSO)** is a population-based optimization algorithm inspired by the collective
behavior of swarms.

In this system:
- Each **particle** represents a possible exam timetable.
- Particles move through the search space by learning from:
  - their **own best solution** (cognitive component), and
  - the **global best solution** found by the swarm (social component).

### How to use this dashboard
- Adjust the parameters on the left to explore different behaviors.
- Click **Run PSO** to generate a timetable.
- The **convergence curve** shows how the best solution improves over iterations.
- Higher accuracy indicates better constraint satisfaction.
- Runtime reflects the trade-off between solution quality and computation time.
""")

with st.expander("📌 Optimization Objectives & Rules", expanded=False):
    st.markdown("""
### 🎯 Optimization Goal
The goal of this system is to **minimize the overall scheduling cost** while satisfying
exam timetabling constraints.

The optimization objective combines:
- **Hard constraint penalties** (must be avoided)
- **Room utilization efficiency** (soft objective)

---

### ❌ Hard Constraints (High Penalty)
Violations of these rules are strongly penalized:

1. **Room Capacity Constraint**  
   - An exam must not be assigned to a room with insufficient seating.
2. **Room-Type Compatibility**
   - Practical exams must be scheduled in **laboratories**.
   - Theory exams should not occupy laboratory rooms.
3. **Room–Timeslot Conflict**
   - A room cannot host more than one exam in the same timeslot.

---

### ⚖️ Soft Objective (Optimization Preference)
These do not invalidate a solution but influence its quality:

- **Room Utilization Balance**
  - Reduce wasted seats by distributing students efficiently across rooms.

---

### 🧮 Fitness Function Overview
The total fitness value is computed as:
**Fitness = Hard Constraint Penalties + Room Utilization Variance**

Lower fitness values indicate better scheduling solutions.
""")

@st.cache_data
def load_data():
    exams = pd.read_csv("exam_timeslot.csv")
    rooms = pd.read_csv("classrooms.csv")
    exams.columns = exams.columns.str.strip().str.lower()
    rooms.columns = rooms.columns.str.strip().str.lower()
    return exams, rooms

exams, rooms = load_data()

# ================= Timeslot Configuration =================
TIMESLOT_LABELS = [
    "09:00 AM", "10:00 AM", "11:00 AM",
    "12:00 PM", "01:00 PM", "02:00 PM",
    "03:00 PM", "04:00 PM"
]

num_timeslots = len(TIMESLOT_LABELS)

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

        total_seats = rooms['capacity'].sum()
        st.write(f"**Total Seats Available:** {total_seats}")

        st.markdown("**Room Capacity**")
        st.write(f"Min: {rooms['capacity'].min()}")
        st.write(f"Max: {rooms['capacity'].max()}")
        st.write(f"Average: {rooms['capacity'].mean():.1f}")

# ================= Sidebar =================
st.sidebar.header("⚙️ Parameters")
num_particles = st.sidebar.slider("Particles", 10, 200, 50, 10)
iterations = st.sidebar.slider("Iterations", 50, 350, 50, 50)
w = st.sidebar.slider(
    "Inertia Weight (w)",
    min_value=0.1,
    max_value=1.2,
    value=0.7,
    step=0.1,
    help="Controls momentum of particles. Higher = more exploration."
)

c1 = st.sidebar.slider(
    "Cognitive Coefficient (c1)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Influence of particle's personal best."
)

c2 = st.sidebar.slider(
    "Social Coefficient (c2)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
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

    # -------- Convergence Speed (numeric) --------
    curve = np.array(result["convergence"])

    start_fitness = curve[0]
    final_fitness = curve[-1]

    # 95% improvement threshold
    target_fitness = final_fitness + 0.05 * (start_fitness - final_fitness)

    convergence_iteration = next(
        (i for i, f in enumerate(curve) if f <= target_fitness),
            len(curve)
    )

    total_seconds = int(result["runtime"])
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    formatted_runtime = f"{minutes} min {seconds:02d} sec"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Best Fitness", f"{result['fitness']:.2f}")
    col2.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
    col3.metric("Runtime", formatted_runtime)
    col4.metric("Convergence Iteration", convergence_iteration)

    # -------- Styled Convergence Curve --------
    st.subheader("📉 Convergence Curve")

    fig, ax = plt.subplots(figsize=(10, 4))

    # Light background (match app)
    fig.patch.set_facecolor("#FFF7DD")
    ax.set_facecolor("#FFFFFF")

    # Plot line
    ax.plot(
         result["convergence"],
        linewidth=2.5,
        color="#80A1BA"  # soft blue from your palette
    )

    # Grid (light, subtle)
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4
    )

    # Axis labels
    ax.set_xlabel("Iterations", color="#24333A")
    ax.set_ylabel("Best Fitness", color="#24333A")

    # Ticks
    ax.tick_params(colors="#24333A")

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#80A1BA")
    ax.spines["bottom"].set_color("#80A1BA")

    st.pyplot(fig)

    # -------- Capacity & Utilization Metrics --------
    solution = result["solution"]

    capacity_violations = 0
    total_wasted_capacity = 0
    total_capacity_used = 0

    for i in range(len(exams)):
        rm = int(np.clip(round(solution[2*i + 1]), 0, len(rooms) - 1))
        students = exams.iloc[i]["num_students"]
        capacity = rooms.iloc[rm]["capacity"]

        if students > capacity:
            capacity_violations += 1
        else:
            total_wasted_capacity += (capacity - students)

        total_capacity_used += students

    wasted_capacity_ratio = (
        total_wasted_capacity / (total_capacity_used + total_wasted_capacity)
        if total_capacity_used > 0 else 0
    )

    col5, col6, col7 = st.columns(3)

    col5.metric("Capacity Violations", capacity_violations)
    col6.metric("Wasted Seats", f"{total_wasted_capacity}")
    col7.metric("Wasted Capacity Ratio", f"{wasted_capacity_ratio*100:.2f}%")


    # -------- Text-style Timetable --------
    st.subheader("🗓️ Final Exam Schedule")

    timeslot_labels = TIMESLOT_LABELS

    days = exams["exam_day"].unique()

    for day in days:
        with st.expander(f"Exam Schedule for Day {day}", expanded=True):

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

            for time in timeslot_labels:
                st.markdown(f"**Time Slot {time}**")
                if not schedule[time]:
                    st.write("  - Free")
                else:
                    for exam in schedule[time]:
                        st.write(f"  - {exam}")
