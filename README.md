Music Similarity Detection with PyTorch and MongoDB
Overview

This project aims to detect similar songs using an Artificial Neural Network (ANN) trained with PyTorch and a dataset stored in MongoDB. The process involves extracting audio features, training the ANN model, and querying MongoDB to find similar songs. The deployment is done using Flask to provide a web interface for users to input a filename and retrieve similar songs.
Technologies Used

    Python: Programming language used for the implementation.
    PyTorch: Deep learning framework for training the ANN model.
    MongoDB: NoSQL database used for storing the audio dataset.
    GridFS: MongoDB specification for storing large files such as audio files.
    Librosa: Python package for audio analysis.
    Mutagen: Python package for reading audio metadata.
    Scikit-learn: Python library for machine learning.
    NumPy: Library for numerical computations in Python.
    Flask: Python web framework used for deployment.
    HTML/CSS: Frontend languages for web interface.
    CUDA: (If applicable) NVIDIA's parallel computing platform for GPU acceleration.

Code Structure

    Data Preparation:
        is_valid_mp3: Function to check the validity of MP3 files.
        extract_metadata: Function to extract metadata from MP3 files.
        compute_features: Function to compute audio features (MFCCs, spectral centroid, zero-crossing rate).
        AudioDataset: Class for creating a PyTorch dataset from MongoDB audio features.
        fetch_features: Function to fetch features for a specific song from MongoDB.
        find_similar_songs: Function to find similar songs using the trained model and MongoDB.

    Model Training:
        ANN: PyTorch ANN model definition.
        Training loop for the ANN model using DataLoader and Adam optimizer.

    Deployment with Flask:
        app.py: Flask application for providing a web interface.
        templates/: HTML templates for web pages.
        static/: Static files (e.g., CSS, JavaScript) for web interface customization.

Usage

    Ensure all required libraries are installed (pip install -r requirements.txt).
    Prepare your audio dataset and MongoDB instance.
    Run the provided scripts to extract features, train the model, and query MongoDB for similar songs.
    Deploy the Flask application using flask run and access the web interface to input filenames and retrieve similar songs.
    Adjust hyperparameters, configurations, and web interface as needed for your specific use case.

Notes

    Ensure your MongoDB instance is properly configured and accessible.
    Monitor the training process and tune hyperparameters for optimal performance.
    Customize the Flask application and frontend templates to suit your project requirements.
    Consider deploying the Flask application on a production server for real-world usage.
