import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model
pipe = pickle.load(open("pipe.pkl", "rb"))

# Set page layout and custom styling
st.set_page_config(page_title="IPL Win Predictor", layout="wide")
st.markdown(
    """
    <style>
    /* Page background and container styling */
    .main {
        background: radial-gradient(circle, #283e51, #0a2342);
        color: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
    }
    /* Title with gradient background */
    .title-container {
        background: linear-gradient(135deg, #a2a7e5, #1f6f8b);
        color: white;
        padding: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
    }
    .title-container img {
        height: 70px;
        margin-right: 15px;
    }
    .title-container h1 {
        font-size: 2.2rem;
        color: white;
    }
     /* Sidebar styling with gradient color and white header */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #34495e, #2c3e50);
        color: #ecf0f1;
    }
    .stButton button {
        background-color: #27ae60;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title with IPL logo and gradient background
st.markdown(
    """
    <div class="title-container">
        <img src="https://thefederal.com/file/2023/01/ipl-logo.webp" alt="IPL Logo">
        <h1>IPL Win Predictor</h1>
    </div>
    """, unsafe_allow_html=True
)

# Sidebar instructions
st.sidebar.header("🏏 Interactive IPL Predictor")
st.sidebar.markdown("""
- Choose match details to predict winning probability.
- Note: Predictions are based on historical data and are for entertainment purposes.
""")

# Team and city selectors with team colors
teams = {
    'Mumbai Indians': '#000080', 'Kolkata Knight Riders': '#800080', 
    'Rajasthan Royals': '#FFC0CB', 'Chennai Super Kings': '#FFFF00', 
    'Sunrisers Hyderabad': '#FFA500', 'Delhi Capitals': '#87CEEB', 
    'Punjab Kings': '#808080', 'Lucknow Super Giants': '#800000', 
    'Gujarat Titans': '#000000', 'Royal Challengers Bengaluru': '#FF0000'
}
cities = [
    'Bangalore', 'Delhi', 'Mumbai', 'Kolkata', 'Hyderabad', 'Chennai',
    'Jaipur', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
    'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
    'Ahmedabad', 'Cuttack', 'Nagpur', 'Visakhapatnam', 'Pune',
    'Raipur', 'Ranchi', 'Abu Dhabi', 'Bengaluru', 'Dubai',
    'Sharjah', 'Navi Mumbai', 'Chandigarh', 'Lucknow', 'Guwahati',
    'Dharamsala', 'Mohali'
]

# Match detail inputs
st.write("### Match Details")
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Select the batting team 🏏", sorted(teams.keys()))
with col2:
    bowling_team = st.selectbox("Select the bowling team 🥎", sorted(teams.keys()))
city = st.selectbox("Select the city 🌆", sorted(cities))

# Live scoring display with target progress
target = st.number_input("Enter Target Score 🎯", step=1)
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input("Enter Current Score 📊", step=1, value=0)
with col4:
    overs = st.number_input("Overs Completed ⏱️", min_value=0, max_value=20)
with col5:
    wickets = st.number_input("Wickets Fallen 🏴", min_value=0, max_value=10)

# Prediction Button
if st.button("Predict Winning Chances 🎉"):
    # Calculate remaining details
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0  # Only calculate crr when overs > 0
    rr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Prediction model input data
    data = {
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "city": [city],
        "runs_left": [runs_left],
        "balls_left": [balls_left],
        "wickets_left": [wickets_left],
        "total_runs_x": [target],
        "crr": [crr],
        "rr": [rr]
    }
    input_df = pd.DataFrame(data)
    result = pipe.predict_proba(input_df)
    loss, win = result[0]

    # Display winning probability
    st.subheader("Winning Probability 🔢")
    st.markdown(f"**{batting_team}**'s Winning Chance: **{round(win * 100)}%**")
    st.markdown(f"**{bowling_team}**'s Winning Chance: **{round(loss * 100)}%**")

    # Bar and Doughnut chart side-by-side
    col_chart1, col_chart2 = st.columns([1, 1])
    
    # Bar chart with team-specific colors
    with col_chart1:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.barh(['Batting Team', 'Bowling Team'], [win * 100, loss * 100], 
                color=[teams[batting_team], teams[bowling_team]])
        for i, v in enumerate([win, loss]):
            ax.text(v * 100 - 10, i, f"{round(v * 100)}%", color='white', va='center', ha='right', weight='bold')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Winning Probability (%)", color='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig)
    
    # Doughnut chart for probability breakdown
    with col_chart2:
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(aspect="equal"))
        wedges, texts, autotexts = ax.pie(
            [win, loss], labels=[f"{batting_team}", f"{bowling_team}"], autopct='%1.1f%%',
            startangle=90, colors=[teams[batting_team], teams[bowling_team]], textprops={'color': 'white'}
        )
        for autotext in autotexts:
            autotext.set_color('black')
        circle = plt.Circle((0, 0), 0.70, color='black', fc='white', linewidth=1.25)
        fig.gca().add_artist(circle)
        plt.title("Winning Probability Breakdown", color="white", fontsize=14)
        st.pyplot(fig)

# Footer with acknowledgment
st.markdown("Developed with ❤️ by Sumanth www.linkedin.com/in/sumanth-godari.")

