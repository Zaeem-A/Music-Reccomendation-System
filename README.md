# Music-Reccomendation-System
Create Your Own Spotify Experience (Project)

Music Similarity Detection with PyTorch and MongoDB
Overview
This project aims to detect similar songs using an Artificial Neural Network (ANN) trained with PyTorch and a dataset stored in MongoDB. The process involves extracting audio features, training the ANN model, and querying MongoDB to find similar songs.
Technologies Used
•	Python: Programming language used for the implementation.
•	PyTorch: Deep learning framework for training the ANN model.
•	MongoDB: NoSQL database used for storing the audio dataset.
•	GridFS: MongoDB specification for storing large files such as audio files.
•	Librosa: Python package for audio analysis.
•	Mutagen: Python package for reading audio metadata.
•	Scikit-learn: Python library for machine learning.
•	NumPy: Library for numerical computations in Python.
•	CUDA: (If applicable) NVIDIA's parallel computing platform for GPU acceleration.
Code Structure
•	Data Preparation:
o	is_valid_mp3: Function to check the validity of MP3 files.
o	extract_metadata: Function to extract metadata from MP3 files.
o	compute_features: Function to compute audio features (MFCCs, spectral centroid, zero-crossing rate).
o	AudioDataset: Class for creating a PyTorch dataset from MongoDB audio features.
o	fetch_features: Function to fetch features for a specific song from MongoDB.
o	find_similar_songs: Function to find similar songs using the trained model and MongoDB.
•	Model Training:
o	ANN: PyTorch ANN model definition.
o	Training loop for the ANN model using DataLoader and Adam optimizer.
Usage
1.	Ensure all required libraries are installed (pip install -r requirements.txt).
2.	Prepare your audio dataset and MongoDB instance.
3.	Run the provided scripts to extract features, train the model, and query MongoDB for similar songs.
4.	Adjust hyperparameters and configurations as needed for your specific use case.
Notes
•	Ensure your MongoDB instance is properly configured and accessible.
•	Adjust the paths and configurations in the code according to your dataset and environment.
•	Monitor the training process and tune hyperparameters for optimal performance.
