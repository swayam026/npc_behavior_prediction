# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("npc_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title("🎮 NPC Action Prediction App")
st.write("Predict NPC actions based on environment and strategy using a trained Decision Tree Classifier.")

# Show accuracy and confusion matrix
st.subheader("📊 Model Evaluation Results")
st.image("confusion_matrix.png", caption="Confusion Matrix of Trained Model")
st.write("**Model Accuracy:** ~91% (approx, varies slightly each run)")
st.markdown("---")

# Collect user input
st.header("🕹️ NPC Input Parameters")

env_state = st.selectbox("Environment State", label_encoders["Environment_State"].classes_)
opp_strategy = st.selectbox("Opponent Strategy", label_encoders["Opponent_Strategy"].classes_)
sensory_input = st.slider("Sensory Input Level", 0.0, 1.0, 0.5)
decision_time = st.slider("Decision Time", 0.0, 3.0, 1.0)
policy_conf = st.slider("Policy Confidence", 0.0, 1.0, 0.5)
reward_score = st.slider("Reward Score", 0, 100, 50)
human_like = st.slider("Human Likeness Score", 0.0, 1.0, 0.5)
behavior_div = st.slider("Behavioral Diversity", 0.0, 1.0, 0.5)

# Convert input to DataFrame
input_data = pd.DataFrame({
    "Environment_State": [env_state],
    "Opponent_Strategy": [opp_strategy],
    "Sensory_Input_Level": [sensory_input],
    "Decision_Time": [decision_time],
    "Policy_Confidence": [policy_conf],
    "Reward_Score": [reward_score],
    "Human_Likeness_Score": [human_like],
    "Behavioral_Diversity": [behavior_div],
})

# Encode categorical inputs
for col in ["Environment_State", "Opponent_Strategy"]:
    le = label_encoders[col]
    input_data[col] = le.transform(input_data[col])

# Predict button
if st.button("Predict NPC Action"):
    prediction = model.predict(input_data)
    predicted_label = target_encoder.inverse_transform(prediction)[0]
    st.success(f"🧠 Predicted NPC Action: **{predicted_label}**")

# ------------------ Footer ------------------
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Made by <b>Swayam Agarwal</b>
    </div>
    """,
    unsafe_allow_html=True
)
