import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pso_core import run_pso
from your_fitness_file import fitness_multiobj  # import your fitness

st.set_page_config(page_title="PSO Exam Scheduling Dashboard", layout="wide")

st.title("📊 PSO-Based University Exam Scheduling")
st.markdown("Interactive performance analysis and multi-objective exploration")

# ===== Load Data =====
@st.cache_data
def load_data():
    exams = pd.read_csv("exam_timeslot.csv")
    rooms = pd.read_csv("classrooms.csv")
    exams.columns = exams.columns.str.lower().str.strip()
    rooms.columns = rooms.columns.str.lower().str.strip()
    return exams, rooms

exams, rooms = load_data()

# ===== Sidebar Controls =====
st.sidebar.header("⚙️ PSO Parameters")

num_particles = st.sidebar.slider("Particles", 10, 60, 30)
iterations = st.sidebar.slider("Iterations", 50, 400, 150)
num_timeslots = st.sidebar.slider("Timeslots per Day", 4, 8, 5)

w = st.sidebar.slider("Inertia Weight (w)", 0.3, 1.2, 0.9)
c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.5, 3.0, 2.0)
c2 = st.sidebar.slider("Social Coefficient (c2)", 0.5, 3.0, 1.2)

run_button = st.sidebar.button("🚀 Run PSO")

# ===== Run PSO =====
if run_button:
    with st.spinner("Running PSO optimization..."):
        results = run_pso(
            exams,
            rooms,
            num_timeslots,
            num_particles,
            iterations,
            w,
            c1,
            c2,
            fitness_multiobj
        )

    st.success("Optimization completed!")

    # ===== Metrics =====
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Fitness", f"{results['best_fitness']:.2f}")
    col2.metric("Constraint Violations", results["violations"])
    col3.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
    col4.metric("Runtime (s)", f"{results['runtime']:.2f}")

    # ===== Convergence Curve =====
    st.subheader("📉 Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(results["convergence"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    ax.set_title("PSO Convergence Behaviour")
    ax.grid(True)
    st.pyplot(fig)

    # ===== Trade-off Insight =====
    st.subheader("⚖️ Objective Trade-off Analysis")
    st.markdown("""
    - Lower fitness indicates fewer violations and better feasibility  
    - Increasing particles improves exploration but increases runtime  
    - Higher inertia weight encourages global exploration  
    - Cognitive/social balance affects convergence speed and stability
    """)

    st.info("Try adjusting parameters to observe accuracy–runtime trade-offs.")
