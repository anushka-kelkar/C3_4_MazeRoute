# C3_4_MazeRoute
AIML miniproject
MazeRout : AI-Powered Terrain-Based Pathfinding Simulation
A* Pathfinding + Machine Learning Terrain Prediction + Pygame Simulation

Overview
This project is an interactive simulation/game that integrates:

A* algorithm for optimal pathfinding
Machine Learning models for predicting terrain type (classification) and difficulty (regression)
Pygame-based maze environment where users can visually explore paths across different terrains The system estimates terrain difficulty using a custom CSV dataset (converted from the Terrain Class Friction Dataset) and dynamically adjusts A* movement cost to choose smarter paths.
Key Features
Terrain Classification (ML Model 1) - Predicts terrain type using features like friction coefficient Models tested: Logistic Regression, SVM, Random Forest Final chosen model: Random Forest Classifier Performance: ~98% accuracy

Terrain Difficulty Prediction (ML Model 2) - Predicts numerical difficulty score for any terrain Regression models used: Linear Regression, Random Forest Regressor, Gradient Boosting Final chosen model: Random Forest Regressor

A* Pathfinding Algorithm - Computes the best path from start to goal Enhanced with terrain-aware movement cost using ML predictions Avoids difficult terrains when possible Fully visualized through the grid

Pygame Maze Simulation Interactive grid-based maze - Start and end points selected by the user Displays path traversal clearly Integrates ML outputs in real-time Users can input custom terrain feature values for prediction

Technical Highlights
Real-time ML inference integrated with Pygame
Terrain-aware cost function inside A*
Modular folder structure
Works with any terrain-style CSV dataset
Lightweight and runs smoothly on most systems
