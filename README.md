project deployed on streamlit : https://npc-action-predictor.streamlit.app/

NPC Action Prediction (Supervised Learning)

This project predicts how a Non-Player Character (NPC) behaves in a game environment using a supervised machine learning model, specifically a Decision Tree Classifier.
The NPC's possible actions are Attack, Defend, Hide, Explore, and Communicate.
The model takes parameters such as environment state, opponent strategy, policy confidence, reward score, decision time, and others to determine which action the NPC is likely to perform.

Initially, the project used a dataset with random or overlapping values that caused low accuracy.
To improve performance, the dataset was logically modified so that each NPC action had distinct behavioral patterns (for example, high confidence and reward for Attack, low confidence and long decision time for Hide).
This improved the model’s accuracy to around 91%.

The original unmodified dataset can be found here:
https://www.kaggle.com/datasets/zara2099/context-aware-npc-behavior-dataset/data

Requirements : Python , pandas , sckit-learn,matplotlib,streamlit,joblib.

Steps to Run

1) Download all project files from this repository and extract them to a folder.

2) Open the folder in your IDE or terminal.

Run the Streamlit app using: streamlit run app.py
Once Streamlit opens in your browser, use the sliders and dropdowns to test different input combinations and see how the NPC's predicted action changes.

Model Information
Algorithm: Decision Tree Classifier
Criterion: Entropy
Max Depth: 5
Model Accuracy: ~91%
Evaluation: Confusion Matrix
