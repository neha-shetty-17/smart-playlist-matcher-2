#!/usr/bin/env python3
"""
Fast Enhanced Smart Playlist Training Script
Optimized for speed while maintaining improved accuracy
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Constants
MOODS = ['happy', 'calm', 'angry', 'sad']
SAMPLE_RATE = 22050
DURATION = 30  # seconds

def extract_emotion_enhanced_features(audio_path):
    """Extract enhanced emotion-specific features efficiently"""
    
    try:
        # Load audio with fixed duration for consistency
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Basic features (same as original)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        # Enhanced features (reduced set for speed)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        rms_energy = librosa.feature.rms(y=y)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Aggregate features
        def aggregate_features(features):
            return [np.mean(features), np.std(features), np.min(features), np.max(features)]
        
        baseline_features = []
        
        # MFCC features
        for i in range(13):
            baseline_features.extend(aggregate_features(mfccs[i]))
        
        # Spectral features
        baseline_features.extend(aggregate_features(spectral_centroids[0]))
        baseline_features.extend(aggregate_features(spectral_rolloff[0]))
        baseline_features.extend(aggregate_features(spectral_bandwidth[0]))
        baseline_features.extend(aggregate_features(zero_crossing_rate[0]))
        
        # Chroma and Tonnetz
        for i in range(12):
            baseline_features.extend(aggregate_features(chroma[i]))
        for i in range(6):
            baseline_features.extend(aggregate_features(tonnetz[i]))
        
        # Enhanced emotion features (reduced)
        emotion_features = []
        
        # Spectral contrast (first 4 bands only)
        for i in range(4):
            emotion_features.extend(aggregate_features(spectral_contrast[i]))
        
        # RMS energy
        emotion_features.extend(aggregate_features(rms_energy[0]))
        
        # Tempo and beat features
        emotion_features.extend([tempo, len(beats), np.std(np.diff(beats)) if len(beats) > 1 else 0])
        
        # Combine all features
        all_features = baseline_features + emotion_features
        
        return np.array(all_features)
        
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None

def create_optimized_ensemble():
    """Create optimized ensemble classifier"""
    
    # Optimized Random Forest
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Light Gradient Boosting
    gb_classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    # Ensemble classifier
    ensemble_classifier = VotingClassifier(
        estimators=[
            ('rf', rf_classifier),
            ('gb', gb_classifier)
        ],
        voting='soft'
    )
    
    return ensemble_classifier

def quick_hyperparameter_tuning(X, y):
    """Quick hyperparameter tuning with reduced search space"""
    
    print("🔍 Performing Quick Hyperparameter Tuning...")
    
    # Reduced parameter grid for speed
    rf_param_grid = {
        'n_estimators': [150, 250],
        'max_depth': [12, 18],
        'min_samples_split': [3, 7],
        'max_features': ['sqrt']
    }
    
    # Create classifier for tuning
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Quick 3-fold CV instead of 5-fold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    from sklearn.model_selection import GridSearchCV
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"📊 Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_enhanced_model_fast():
    """Train enhanced model with optimizations for speed"""
    
    print("🚀 Smart Playlist Matcher - FAST ENHANCED Training")
    print("=" * 60)
    
    # Dataset paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    audio_dir = "datasets/deam/DEAM/audio"
    valence_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
    arousal_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/dynamic_annotations_averaged_songs_1-2000/dynamic_annotations_averaged_songs_1_2000.csv"
    
    # Load original data with enhanced features
    print("🔄 Loading data with enhanced features...")
    X_original = []
    y_original = []
    
    # Load annotations
    valence_file = ROOT_DIR / valence_file
    arousal_file = ROOT_DIR / arousal_file
    
    valence_df = pd.read_csv(valence_file)
    arousal_df = pd.read_csv(arousal_file)
    
    # Calculate mean arousal from dynamic data
    arousal_cols = [col for col in arousal_df.columns if 'arousal' in col and col != 'song_id']
    arousal_df['arousal_mean'] = arousal_df[arousal_cols].mean(axis=1)
    
    valence_threshold = valence_df[' valence_mean'].median()
    arousal_threshold = arousal_df['arousal_mean'].median()
    
    # Process each audio file
    audio_path = ROOT_DIR / audio_dir
    processed_count = 0
    
    for song_id in range(2, min(502, 2002)):  # Process first 500 songs for speed
        if processed_count >= 400:  # Limit to 400 samples for training
            break
            
        mp3_file = audio_path / f"{song_id}.mp3"
        
        if not mp3_file.exists():
            continue
        
        # Extract enhanced features
        features = extract_emotion_enhanced_features(mp3_file)
        if features is None:
            continue
        
        # Get annotations
        valence_row = valence_df[valence_df['song_id'] == song_id]
        arousal_row = arousal_df[arousal_df['song_id'] == song_id]
        
        if not valence_row.empty and not arousal_row.empty:
            valence = valence_row[' valence_mean'].iloc[0]
            arousal = arousal_row['arousal_mean'].iloc[0]
            
            # Determine mood
            if valence >= valence_threshold and arousal >= arousal_threshold:
                mood = 'happy'
            elif valence >= valence_threshold and arousal < arousal_threshold:
                mood = 'calm'
            elif valence < valence_threshold and arousal >= arousal_threshold:
                mood = 'angry'
            else:
                mood = 'sad'
            
            X_original.append(features)
            y_original.append(mood)
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"  Processed {processed_count} samples...")
    
    print(f"📊 Original samples: {len(X_original)}")
    print(f"📏 Feature dimension: {len(X_original[0])}")
    
    # Convert to numpy arrays
    X = np.array(X_original)
    y = np.array(y_original)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("🎯 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Quick hyperparameter tuning
    best_rf = quick_hyperparameter_tuning(X_train_scaled, y_train)
    
    # Create ensemble with optimized RF
    ensemble = create_optimized_ensemble()
    
    # Replace RF in ensemble with best one
    ensemble.estimators[0] = ('rf', best_rf)
    
    # Train ensemble
    print("🎯 Training optimized ensemble...")
    ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("📊 Evaluating enhanced model...")
    y_pred = ensemble.predict(X_test_scaled)
    
    print(f"\n📈 Enhanced Model Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Cross-validation
    print("🔄 Performing cross-validation...")
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=3, scoring='f1_weighted')
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model
    print("💾 Saving enhanced model...")
    model_dir = ROOT_DIR / "app" / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_data = {
        'model': ensemble,
        'scaler': scaler,
        'features': len(X_original[0]),
        'moods': MOODS,
        'model_type': 'enhanced_fast',
        'samples': len(X_original)
    }
    
    model_path = model_dir / "smart_playlist_classifier_enhanced_fast.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Enhanced model saved to: {model_path}")
    print(f"📏 Feature count: {len(X_original[0])}")
    
    # Test prediction
    print(f"\n🧪 Testing enhanced prediction...")
    test_audio = ROOT_DIR / audio_dir / "1.mp3"
    if test_audio.exists():
        try:
            features = extract_emotion_enhanced_features(test_audio)
            if features is not None:
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction = ensemble.predict(features_scaled)[0]
                probabilities = ensemble.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
                
                print(f"🎵 Enhanced prediction:")
                print(f"  Mood: {prediction} (confidence: {confidence:.2f})")
                print(f"  Probabilities: {dict(zip(MOODS, probabilities))}")
        except Exception as e:
            print(f"❌ Test prediction failed: {e}")
    
    print(f"\n🎉 FAST Enhanced Training Complete!")

if __name__ == "__main__":
    train_enhanced_model_fast()
