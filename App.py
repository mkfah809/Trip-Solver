#pip install streamlit 
#pip install pandas 
#pip install numpy 
#pip install matplotlib 
#pip install haversine python-tsp



import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine, Unit
from python_tsp.exact import solve_tsp_dynamic_programming

# Page config must be FIRST
st.set_page_config(
    page_title="TSP Route Optimizer", 
    layout="wide"
)

# -----------------------------
# Cached functions
# -----------------------------
@st.cache_data
def load_cities_data(_csv_path="uscitiesDataSet.csv"):
    return pd.read_csv(_csv_path)

@st.cache_data
def build_distance_matrix(_coords):
    num_cities = len(_coords)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist_matrix[i, j] = haversine(_coords[i], _coords[j], unit=Unit.MILES)
    return dist_matrix

# -----------------------------
# Main app
# -----------------------------
def main():
    st.title("TSP Route Optimizer")
    st.markdown("**Optimize the perfect city tour!** Choose cities and start point.")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        random_seed = st.number_input("Random Seed", 0, 10000, 42)
        num_cities = st.slider("Number of Cities", 5, 25, 8)
        
        st.divider()
        if st.button("New Route", use_container_width=True):
            st.rerun()
    
    # Load data
    try:
        df = load_cities_data()
        max_cities = min(num_cities, len(df))
        
        # Reproducible random selection
        random.seed(random_seed)
        np.random.seed(random_seed)
        cities_df = df.sample(n=max_cities, random_state=random_seed).reset_index(drop=True)
        
        cities = cities_df["city"].tolist()
        coords = cities_df[["lat", "lng"]].values.tolist()
        
        st.sidebar.info(f"{len(cities)} cities loaded")
        
        # Start city
        start_city = st.selectbox("Start City", cities, 0)
        start_idx = cities.index(start_city)
        
    except FileNotFoundError:
        st.error("**uscitiesDataSet.csv** missing!")
        st.stop()
    
    # Compute button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Solve TSP", type="primary"):
            with st.spinner("Computing optimal route..."):
                # Solve
                dist_matrix = build_distance_matrix(coords)
                permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
                
                # Rotate to start city
                start_pos = permutation.index(start_idx)
                route = permutation[start_pos:] + permutation[:start_pos]
                route.append(route[0])  # Close loop
                
                # Store results
                optimal_cities = [cities[i] for i in route]
                optimal_coords = [coords[i] for i in route]
                
                st.session_state.results = {
                    'cities': optimal_cities,
                    'coords': optimal_coords,
                    'distance': distance,
                    'num_cities': len(route) - 1
                }
    
    # Results
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Distance", f"{results['distance']:.0f} miles")
        col2.metric("Cities", results['num_cities'])
        col3.metric("Start", results['cities'][0])
        
        # Route list
        st.subheader("Route Order")
        for i, (city1, city2) in enumerate(zip(results['cities'][:-1], results['cities'][1:])):
            st.write(f"**{i+1}.** {city1} → {city2}")
        
        # Plot
        st.subheader("Route Map")
        fig, ax = plt.subplots(figsize=(12, 9))
        
        lons, lats = np.array(results['coords']).T
        ax.plot(lons, lats, 'r-', linewidth=3, label='Route')
        ax.scatter(lons, lats, c='blue', s=120, zorder=5, label='Cities')
        
        # Labels (unique cities only)
        for i in range(len(results['cities']) - 1):
            ax.annotate(f"({i+1}) {results['cities'][i]}", 
                       (lons[i], lats[i]), fontsize=9,
                       ha='right', va='bottom',
                       bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.8))
        
        ax.scatter(lons[0], lats[0], c='green', s=250, marker='*', 
                  zorder=10, label='Start/End', edgecolors='black')
        
        ax.set_title("Optimal TSP Route", fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Clear
        if st.button("Clear Results"):
            del st.session_state.results
            st.rerun()

if __name__ == "__main__":
    main()
