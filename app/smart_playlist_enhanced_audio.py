#!/usr/bin/env python3
"""
🎵 Smart Playlist Matcher - Enhanced with Audio Playback
Full dataset processing with enhanced model and audio playback
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import pickle
import warnings
import time
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from train_enhanced_model_fast import extract_emotion_enhanced_features

app = FastAPI(title="Enhanced Smart Playlist Matcher with Audio", description="Upload audio for mood prediction and recommendations with playback")

# Global variables
model_data = None
feature_database = None
audio_metadata = None
model_path = None  # Add global model path

class PredictionResponse(BaseModel):
    mood: str
    confidence: float
    mood_percentages: dict  # All mood classes with percentages
    tempo: dict
    loudness: dict
    recommendations: list

class RecommendationItem(BaseModel):
    filename: str
    mood: str
    confidence: float
    similarity_score: float
    tempo: dict
    loudness: dict
    audio_path: str

def extract_improved_mood_features(audio_path):
    """Extract enhanced features for improved mood classification"""
    try:
        # Extract base features
        base_features = extract_emotion_enhanced_features(audio_path)
        if base_features is None:
            return None
        
        # Extract additional features for improved classification
        y, sr = librosa.load(audio_path, duration=30)
        
        # Tempo features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_mean = float(tempo)
        tempo_std = float(np.std(beats)) if len(beats) > 1 else 0.0
        
        # Energy features
        rms = librosa.feature.rms(y=y)
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))
        
        # Create enhanced feature vector
        enhanced_features = np.concatenate([
            base_features,  # Original 163 features
            [tempo_mean, tempo_std, energy_mean, energy_std]  # Additional 4 features
        ])
        
        return enhanced_features
        
    except Exception as e:
        print(f"⚠️  Error extracting improved features: {e}")
        return None

def load_model_and_database():
    """Load enhanced model and build feature database efficiently"""
    global model_data, feature_database, audio_metadata, model_path
    
    print("🔄 Loading enhanced model and building feature database...")
    start_time = time.time()
    
    # Load improved mood classifier
    model_path = Path(__file__).resolve().parent / "models" / "improved_mood_classifier.pkl"
    
    if not model_path.exists():
        # Fallback to original enhanced model if improved model doesn't exist
        model_path = Path(__file__).resolve().parent / "models" / "smart_playlist_classifier_enhanced_fast.pkl"
        print("⚠️  Using original enhanced model (improved model not found)")
    else:
        print("✅ Using improved mood classifier with enhanced features")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Enhanced model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✅ Enhanced model loaded successfully!")
    print(f"📏 Feature count: {model_data['features']}")
    print(f"🎯 Model type: {model_data.get('model_type', 'unknown')}")
    
    # Build feature database from dataset
    ROOT_DIR = Path(__file__).resolve().parent.parent
    audio_dir = ROOT_DIR / "datasets/deam/DEAM/audio"
    valence_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
    arousal_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/dynamic_annotations_averaged_songs_1-2000/dynamic_annotations_averaged_songs_1_2000.csv"
    
    # Load annotations for mood labels
    print("📊 Loading annotations...")
    valence_df = pd.read_csv(ROOT_DIR / valence_file)
    arousal_df = pd.read_csv(ROOT_DIR / arousal_file)
    
    # Clean column names
    valence_df.columns = valence_df.columns.str.strip()
    arousal_df.columns = arousal_df.columns.str.strip()
    
    # Calculate mean arousal from dynamic data
    arousal_cols = [col for col in arousal_df.columns if 'arousal' in col and col != 'song_id']
    arousal_df['arousal_mean'] = arousal_df[arousal_cols].mean(axis=1)
    
    # Merge annotations
    annotations = pd.merge(valence_df, arousal_df, on='song_id', suffixes=('', '_y'))
    
    # Convert to mood labels using enhanced classification
    def valence_arousal_to_mood(valence, arousal, tempo_energy=None):
        # Use dataset-specific medians for better classification
        valence_threshold = 4.9  # Dataset median
        arousal_threshold = 4.9  # Dataset median
        
        # Base classification on valence-arousal
        if valence >= valence_threshold and arousal >= arousal_threshold:
            base_mood = 'happy'
        elif valence >= valence_threshold and arousal < arousal_threshold:
            base_mood = 'calm'
        elif valence < valence_threshold and arousal >= arousal_threshold:
            base_mood = 'angry'
        else:
            base_mood = 'sad'
        
        # Enhanced classification: consider high energy for angry classification
        if tempo_energy is not None and tempo_energy > 0.7:  # High energy threshold
            if base_mood == 'calm':
                return 'angry'  # High energy + positive valence = angry/energetic
            elif base_mood == 'sad':
                return 'angry'  # High energy + negative valence = angry
        
        return base_mood
    
    # Use correct column names after merge
    valence_col = 'valence_mean' if 'valence_mean' in annotations.columns else 'valence_mean_y'
    arousal_col = 'arousal_mean' if 'arousal_mean' in annotations.columns else 'arousal_mean_y'
    
    annotations['mood'] = annotations.apply(
        lambda row: valence_arousal_to_mood(row[valence_col], row[arousal_col]), 
        axis=1
    )
    
    print(f"📋 Loaded {len(annotations)} annotations")
    
    # Extract features for all audio files
    features_list = []
    metadata_list = []
    
    print("🔧 Building feature database (processing all songs)...")
    
    processed_count = 0
    for idx, row in annotations.iterrows():
        song_id = row['song_id']
        audio_path = audio_dir / f"{song_id}.mp3"
        
        if audio_path.exists():
            try:
                features = extract_emotion_enhanced_features(audio_path)
                if features is not None:
                    # Check feature dimensions
                    expected_features = model_data['features']
                    if len(features) != expected_features:
                        if len(features) > expected_features:
                            features = features[:expected_features]
                        else:
                            features = np.pad(features, (0, expected_features - len(features)), 'constant')
                    
                    # Scale features using enhanced model scaler
                    features_scaled = model_data['scaler'].transform(features.reshape(1, -1))[0]
                    
                    features_list.append(features_scaled)
                    metadata_list.append({
                        'filename': f"{song_id}.mp3",
                        'song_id': song_id,
                        'mood': row['mood'],
                        'valence': row[valence_col],
                        'arousal': row[arousal_col],
                        'audio_path': str(audio_path)
                    })
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        elapsed = time.time() - start_time
                        speed = processed_count / elapsed
                        eta = (len(annotations) - processed_count) / speed
                        print(f"  Processed {processed_count}/{len(annotations)} songs ({speed:.1f} songs/sec, ETA: {eta:.0f}s)")
                        
            except Exception as e:
                print(f"⚠️  Error processing {song_id}: {e}")
                continue
    
    if features_list:
        feature_database = np.array(features_list)
        audio_metadata = pd.DataFrame(metadata_list)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Built database with {len(feature_database)} songs in {processing_time:.1f} seconds")
        print(f"⚡ Processing speed: {len(feature_database)/processing_time:.1f} songs/second")
        print(f"💾 Database size: {feature_database.nbytes/1024/1024:.1f} MB")
    else:
        raise ValueError("No features could be extracted from dataset")

def find_similar_songs(query_features, top_k=5):
    """Find most similar songs using cosine similarity"""
    if feature_database is None:
        return []
    
    # Calculate similarity scores
    similarities = cosine_similarity([query_features], feature_database)[0]
    
    # Get top-k most similar songs
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    recommendations = []
    for idx in top_indices:
        song_data = audio_metadata.iloc[idx].to_dict()
        
        # Extract engineered features for display
        original_features = model_data['scaler'].inverse_transform([feature_database[idx]])[0]
        # For enhanced model, tempo and energy features are at the end
        if len(original_features) >= 6:
            tempo_mean = float(original_features[-6]) if len(original_features) >= 6 else 120
            tempo_std = float(original_features[-5]) if len(original_features) >= 5 else 10
            tempo_cv = float(original_features[-4]) if len(original_features) >= 4 else 0.1
            loudness_mean = float(original_features[-3]) if len(original_features) >= 3 else 0.5
            loudness_std = float(original_features[-2]) if len(original_features) >= 2 else 0.1
            loudness_cv = float(original_features[-1]) if len(original_features) >= 1 else 0.2
        else:
            # Fallback values
            tempo_mean = tempo_std = tempo_cv = 120
            loudness_mean = loudness_std = loudness_cv = 0.5
        
        # Predict mood confidence for this song using enhanced model
        mood_probs = model_data['model'].predict_proba([feature_database[idx]])[0]
        mood_idx = np.argmax(mood_probs)
        confidence = mood_probs[mood_idx]
        
        recommendations.append({
            'filename': song_data['filename'],
            'mood': song_data['mood'],
            'confidence': float(confidence),
            'similarity_score': float(similarities[idx]),
            'tempo': {
                'mean_bpm': round(tempo_mean, 2),
                'variability': round(tempo_std, 2),
                'coefficient_of_variation': round(tempo_cv, 3)
            },
            'loudness': {
                'mean_energy': round(loudness_mean, 4),
                'variability': round(loudness_std, 4),
                'coefficient_of_variation': round(loudness_cv, 3)
            },
            'audio_path': f"/audio/{song_data['filename']}"
        })
    
    return recommendations

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    try:
        load_model_and_database()
        print("🚀 Enhanced Smart Playlist Matcher with Audio is ready!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Main web interface with audio playback"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎵 Enhanced Smart Playlist Matcher with Audio</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .upload-section {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .upload-area {
            border: 3px dashed #fff;
            border-radius: 10px;
            padding: 40px;
            background: rgba(255, 255, 255, 0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        
        .upload-area h3 {
            color: white;
            margin-bottom: 10px;
            font-size: 1.5em;
        }
        
        .upload-area p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1em;
        }
        
        #fileInput {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        
        .results {
            display: none;
        }
        
        .prediction-result {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            color: white;
        }
        
        .mood-display {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }
        
        .mood-analysis {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
        }
        
        .mood-item {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        
        .mood-label {
            width: 80px;
            font-weight: bold;
            text-transform: capitalize;
        }
        
        .mood-bar-container {
            flex: 1;
            height: 25px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            margin: 0 10px;
            position: relative;
            overflow: hidden;
        }
        
        .mood-bar {
            height: 100%;
            border-radius: 12px;
            transition: width 0.5s ease;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        }
        
        .mood-percentage {
            width: 50px;
            text-align: right;
            font-weight: bold;
        }
        
        .confidence-score {
            text-align: center;
            font-size: 1.2em;
            margin: 10px 0;
            padding: 10px;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 8px;
        }
        
        .audio-player {
            width: 100%;
            margin: 10px 0;
            border-radius: 5px;
        }
        
        .audio-player-section {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        
        .feature-box {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
        }
        
        .recommendations {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .recommendations h3 {
            color: #333;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .song-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .song-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .song-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }
        
        .song-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        
        .song-mood {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        
        .song-stats {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }
        
        .similarity-score {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-align: center;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            font-size: 1.2em;
            color: #666;
        }
        
        .error {
            background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        
        .enhanced-badge {
            background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .audio-badge {
            background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Smart Playlist Matcher</h1>
        <p class="subtitle">
            Upload audio for mood prediction and recommendations
            <span class="enhanced-badge">ENHANCED</span>
            <span class="audio-badge">AUDIO PLAYBACK</span>
        </p>
        
        <div class="upload-section">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <h3>📁 Upload Audio File</h3>
                <p>Click to browse or drag & drop your MP3 file here</p>
                <p><small>Enhanced with 163 features • Full dataset (~1700+ songs) • Audio playback included</small></p>
            </div>
            <input type="file" id="fileInput" accept="audio/*" onchange="handleFileUpload()">
        </div>
        
        <div id="results" class="results">
            <div class="prediction-result">
                <div class="mood-display" id="moodDisplay">-</div>
                
                <!-- Multi-Class Mood Analysis -->
                <div class="mood-analysis">
                    <h4 style="text-align: center; margin-bottom: 15px;">🎭 Mood Analysis</h4>
                    <div id="moodPercentages">
                        <div class="loading">Analyzing mood components...</div>
                    </div>
                    <div class="confidence-score" id="confidenceScore">
                        Confidence: -
                    </div>
                </div>
                
                <!-- Uploaded Audio Player -->
                <div class="audio-player-section">
                    <h4>🎵 Your Uploaded Audio</h4>
                    <audio id="uploadedAudio" controls class="audio-player" style="width: 100%; margin: 10px 0;">
                        Your browser does not support the audio element.
                    </audio>
                    <div id="uploadedFileName" style="color: #666; font-size: 0.9em; margin-bottom: 15px;">No file uploaded</div>
                </div>
                
                <div class="features-grid">
                    <div class="feature-box">
                        <h4>🎵 Tempo</h4>
                        <div><strong>Mean BPM:</strong> <span id="tempoMean">-</span></div>
                        <div><strong>Variability:</strong> <span id="tempoVar">-</span></div>
                        <div><strong>CV:</strong> <span id="tempoCV">-</span></div>
                    </div>
                    <div class="feature-box">
                        <h4>🔊 Loudness</h4>
                        <div><strong>Mean Energy:</strong> <span id="loudnessMean">-</span></div>
                        <div><strong>Variability:</strong> <span id="loudnessVar">-</span></div>
                        <div><strong>CV:</strong> <span id="loudnessCV">-</span></div>
                    </div>
                </div>
            </div>
            
            <div class="recommendations">
                <h3>🎧 Similar Songs Recommendations</h3>
                <div id="songGrid" class="song-grid"></div>
            </div>
        </div>
    </div>
    
    <script>
        async function handleFileUpload() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) return;
            
            // Show loading
            document.getElementById('results').style.display = 'block';
            document.getElementById('moodDisplay').innerHTML = '🔄 Analyzing...';
            document.getElementById('songGrid').innerHTML = '<div class="loading">Processing audio and finding recommendations...</div>';
            
            // Set up uploaded audio player
            const uploadedAudio = document.getElementById('uploadedAudio');
            const uploadedFileName = document.getElementById('uploadedFileName');
            
            // Create object URL for the uploaded file
            const audioUrl = URL.createObjectURL(file);
            uploadedAudio.src = audioUrl;
            uploadedFileName.textContent = `📁 ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
            
            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                
                // Update prediction result
                document.getElementById('moodDisplay').innerHTML = `🎵 ${result.mood}`;
                
                // Update mood percentages with visual bars
                const moodPercentagesDiv = document.getElementById('moodPercentages');
                moodPercentagesDiv.innerHTML = '';
                
                // Sort moods by percentage for better visualization
                const sortedMoods = Object.entries(result.mood_percentages)
                    .sort((a, b) => b[1] - a[1]);
                
                sortedMoods.forEach(([mood, percentage]) => {
                    const moodItem = document.createElement('div');
                    moodItem.className = 'mood-item';
                    
                    const percentageValue = (percentage * 100).toFixed(1);
                    const barWidth = percentage * 100;
                    
                    // Different colors for different moods
                    const moodColors = {
                        'happy': '#ff6b6b',
                        'sad': '#4ecdc4', 
                        'energetic': '#ff9f43',
                        'calm': '#45b7d1'
                    };
                    
                    const barColor = moodColors[mood] || '#96ceb4';
                    
                    moodItem.innerHTML = `
                        <div class="mood-label">${mood}</div>
                        <div class="mood-bar-container">
                            <div class="mood-bar" style="width: 0%; background: ${barColor};"></div>
                        </div>
                        <div class="mood-percentage">${percentageValue}%</div>
                    `;
                    
                    moodPercentagesDiv.appendChild(moodItem);
                    
                    // Animate the bar
                    setTimeout(() => {
                        moodItem.querySelector('.mood-bar').style.width = `${barWidth}%`;
                    }, 100);
                });
                
                // Update confidence score
                const confidencePercentage = (result.confidence * 100).toFixed(1);
                document.getElementById('confidenceScore').innerHTML = 
                    `🎯 Confidence: ${confidencePercentage}%`;
                
                // Update tempo features
                document.getElementById('tempoMean').textContent = result.tempo.mean_bpm;
                document.getElementById('tempoVar').textContent = result.tempo.variability;
                document.getElementById('tempoCV').textContent = result.tempo.coefficient_of_variation;
                
                // Update loudness features
                document.getElementById('loudnessMean').textContent = result.loudness.mean_energy;
                document.getElementById('loudnessVar').textContent = result.loudness.variability;
                document.getElementById('loudnessCV').textContent = result.loudness.coefficient_of_variation;
                
                // Update recommendations
                const songGrid = document.getElementById('songGrid');
                songGrid.innerHTML = '';
                
                result.recommendations.forEach((song, index) => {
                    const songCard = document.createElement('div');
                    songCard.className = 'song-card';
                    songCard.innerHTML = `
                        <div class="song-title">${song.filename}</div>
                        <div class="song-mood">${song.mood}</div>
                        
                        <!-- Audio Player for Recommendation -->
                        <audio controls class="audio-player" preload="none">
                            <source src="${song.audio_path}" type="audio/mpeg">
                            Your browser does not support the audio element.
                        </audio>
                        
                        <div class="song-stats">
                            <div><strong>Tempo:</strong> ${song.tempo.mean_bpm} BPM</div>
                            <div><strong>Tempo Var:</strong> ${song.tempo.variability}</div>
                            <div><strong>Energy:</strong> ${song.loudness.mean_energy}</div>
                            <div><strong>Energy Var:</strong> ${song.loudness.variability}</div>
                        </div>
                        <div class="similarity-score">
                            ${(song.similarity_score * 100).toFixed(1)}% Similar
                        </div>
                    `;
                    songGrid.appendChild(songCard);
                });
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('moodDisplay').innerHTML = '❌ Error occurred';
                document.getElementById('songGrid').innerHTML = '<div class="error">Failed to analyze audio file. Please try again.</div>';
            }
        }
        
        // Drag and drop functionality
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(255, 255, 255, 0.3)';
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(255, 255, 255, 0.1)';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(255, 255, 255, 0.1)';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('fileInput').files = files;
                handleFileUpload({ target: { files: files } });
            }
        });
        
        // Handle file input change for audio player
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const uploadedAudio = document.getElementById('uploadedAudio');
                const uploadedFileName = document.getElementById('uploadedFileName');
                
                // Create object URL for the uploaded file
                const audioUrl = URL.createObjectURL(file);
                uploadedAudio.src = audioUrl;
                uploadedFileName.textContent = `📁 ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
            }
        });
    </script>
</body>
</html>
"""

@app.post("/predict", response_model=PredictionResponse)
async def predict_and_recommend(file: UploadFile = File(...)):
    """Predict mood and recommend similar songs"""
    try:
        print(f"📁 Received file: {file.filename}")
        print(f"📄 Content type: {file.content_type}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Please upload an audio file")
        
        # Save uploaded file temporarily
        temp_path = Path("temp_audio.mp3")
        print(f"💾 Saving to: {temp_path}")
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"✅ File saved, size: {temp_path.stat().st_size if temp_path.exists() else 0} bytes")
        
        try:
            # Extract features from uploaded audio
            print("🔧 Extracting features...")
            # Use improved feature extraction for better mood classification
            if 'improved_mood_classifier' in str(model_path):
                features = extract_improved_mood_features(temp_path)
                print("🎯 Using improved mood classifier with enhanced features")
            else:
                features = extract_emotion_enhanced_features(temp_path)
                print("🎯 Using original enhanced model")
                
            if features is None:
                print("❌ Feature extraction returned None")
                raise HTTPException(status_code=400, detail="Could not extract features from audio file")
            
            print(f"✅ Features extracted, shape: {features.shape if hasattr(features, 'shape') else 'No shape'}")
            
            # Check feature dimensions
            expected_features = model_data['features']
            print(f"🎯 Expected features: {expected_features}")
            
            if len(features) != expected_features:
                print(f"⚠️ Feature dimension mismatch: got {len(features)}, expected {expected_features}")
                # Adjust features if needed
                if len(features) > expected_features:
                    features = features[:expected_features]
                    print(f"✂️ Trimmed features to: {len(features)}")
                else:
                    features = np.pad(features, (0, expected_features - len(features)), 'constant')
                    print(f"➕ Padded features to: {len(features)}")
            
            print(f"🔧 Scaling features...")
            # Scale features using enhanced model scaler
            features_scaled = model_data['scaler'].transform(features.reshape(1, -1))[0]
            print(f"✅ Features scaled, shape: {features_scaled.shape}")
            
            print(f"🎯 Making prediction...")
            # Predict mood using enhanced model
            mood = model_data['model'].predict([features_scaled])[0]
            mood_probabilities = model_data['model'].predict_proba([features_scaled])[0]
            confidence = np.max(mood_probabilities)
            
            # Get all mood classes and their probabilities
            mood_classes = model_data['model'].classes_
            mood_percentages = {}
            for i, mood_class in enumerate(mood_classes):
                mood_percentages[mood_class] = float(mood_probabilities[i])
            
            print(f"✅ Prediction: {mood} with confidence: {confidence}")
            print(f"📊 All mood percentages: {mood_percentages}")
            
            # Extract engineered features for response (enhanced model has different feature structure)
            # For enhanced model, tempo and energy features are at the end
            if len(features) >= 6:
                tempo_mean = float(features[-6]) if len(features) >= 6 else 120
                tempo_std = float(features[-5]) if len(features) >= 5 else 10
                tempo_cv = float(features[-4]) if len(features) >= 4 else 0.1
                loudness_mean = float(features[-3]) if len(features) >= 3 else 0.5
                loudness_std = float(features[-2]) if len(features) >= 2 else 0.1
                loudness_cv = float(features[-1]) if len(features) >= 1 else 0.2
            else:
                # Fallback values
                tempo_mean = tempo_std = tempo_cv = 120
                loudness_mean = loudness_std = loudness_cv = 0.5
            
            print(f"🎵 Tempo: mean={tempo_mean}, std={tempo_std}, cv={tempo_cv}")
            print(f"🔊 Loudness: mean={loudness_mean}, std={loudness_std}, cv={loudness_cv}")
            
            print("🔍 Finding similar songs...")
            # Find similar songs
            recommendations = find_similar_songs(features_scaled, top_k=5)
            print(f"✅ Found {len(recommendations)} recommendations")
            
            result = PredictionResponse(
                mood=mood,
                confidence=float(confidence),
                mood_percentages=mood_percentages,
                tempo={
                    'mean_bpm': round(tempo_mean, 2),
                    'variability': round(tempo_std, 2),
                    'coefficient_of_variation': round(tempo_cv, 3)
                },
                loudness={
                    'mean_energy': round(loudness_mean, 4),
                    'variability': round(loudness_std, 4),
                    'coefficient_of_variation': round(loudness_cv, 3)
                },
                recommendations=recommendations
            )
            
            print(f"✅ Result prepared successfully")
            return result
            
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
                print(f"🗑️ Cleaned up temp file")
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ DETAILED ERROR in prediction: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Mount static files for audio serving
app.mount("/audio", StaticFiles(directory="c:/Users/HARSHITH KUMAR/Desktop/ml/new_smart_playlist_match/datasets/deam/DEAM/audio"), name="audio")

if __name__ == "__main__":
    uvicorn.run("smart_playlist_enhanced_audio:app", host="0.0.0.0", port=8002, reload=True)
