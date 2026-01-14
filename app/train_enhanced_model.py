#!/usr/bin/env python3
"""
Enhanced Model Training with Hyperparameter Tuning and Advanced Features
"""

import sys
import numpy as np
import pandas as pd
import pickle
import librosa
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from train_complete_smart_playlist import SmartPlaylistClassifier, augment_audio, MOODS

def extract_emotion_enhanced_features(audio_path, duration=30):
    """Extract enhanced features for better emotion detection"""
    
    try:
        y, sr = librosa.load(str(audio_path), duration=duration)
        
        # Basic features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        # Enhanced emotion-specific features
        # 1. Spectral contrast (distinguishes calm vs angry)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # 2. RMS energy (distinguishes calm vs energetic)
        rms = librosa.feature.rms(y=y)
        
        # 3. Tempo and onset detection (distinguishes happy vs sad)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        
        # 4. Harmonic-percussive separation (distinguishes calm vs angry)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # 5. Mel-frequency cepstral delta features (captures emotion dynamics)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Aggregate statistics
        def aggregate_features(features):
            return np.concatenate([
                np.mean(features, axis=1),
                np.std(features, axis=1),
                np.max(features, axis=1),
                np.min(features, axis=1)
            ])
        
        # Original features
        baseline_features = aggregate_features(np.vstack([
            mfccs, spectral_centroids, spectral_rolloff, 
            spectral_bandwidth, zero_crossing_rate, chroma, tonnetz
        ]))
        
        # Enhanced emotion features
        emotion_features = aggregate_features(np.vstack([
            spectral_contrast, rms, onset_strength.reshape(1, -1)
        ]))
        
        # Harmonic vs percussive ratio (distinguishes calm vs angry)
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        hp_ratio = harmonic_energy / (percussive_energy + 1e-8)
        
        # Tempo variability (distinguishes happy vs sad)
        if len(beats) > 1:
            beat_intervals = librosa.frames_to_time(beats, sr=sr)
            tempo_variability = np.std(np.diff(beat_intervals)) if len(beat_intervals) > 1 else 0
        else:
            tempo_variability = 0
        
        # Onset density (distinguishes energetic vs calm)
        onset_density = len(onset_frames) / len(y) if len(y) > 0 else 0
        
        # RMS variability (distinguishes angry vs calm)
        rms_variability = np.std(rms)
        
        # Spectral contrast variability (distinguishes happy vs sad)
        contrast_variability = np.std(spectral_contrast)
        
        # Engineered emotion features
        engineered_features = np.array([
            tempo, tempo_variability, hp_ratio, onset_density,
            rms_variability, contrast_variability,
            np.mean(rms), np.std(rms),  # RMS statistics
            np.mean(spectral_contrast), np.std(spectral_contrast)  # Contrast statistics
        ])
        
        # Combine all features
        all_features = np.concatenate([
            baseline_features,  # Original 63 features
            emotion_features,   # Enhanced emotion features
            engineered_features  # Engineered emotion features
        ])
        
        return all_features
        
    except Exception as e:
        print(f" Error extracting features from {audio_path}: {e}")
        return None

def extract_features_with_engineered_array(y, sr):
    """Extract features from audio array (for augmentation)"""
    
    try:
        # Basic features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        # Enhanced emotion-specific features
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Delta features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Aggregate statistics
        def aggregate_features(features):
            return np.concatenate([
                np.mean(features, axis=1),
                np.std(features, axis=1),
                np.max(features, axis=1),
                np.min(features, axis=1)
            ])
        
        # Original features
        baseline_features = aggregate_features(np.vstack([
            mfccs, spectral_centroids, spectral_rolloff, 
            spectral_bandwidth, zero_crossing_rate, chroma, tonnetz
        ]))
        
        # Enhanced emotion features
        emotion_features = aggregate_features(np.vstack([
            spectral_contrast, rms, onset_strength.reshape(1, -1)
        ]))
        
        # Engineered features
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        hp_ratio = harmonic_energy / (percussive_energy + 1e-8)
        
        if len(beats) > 1:
            beat_intervals = librosa.frames_to_time(beats, sr=sr)
            tempo_variability = np.std(np.diff(beat_intervals)) if len(beat_intervals) > 1 else 0
        else:
            tempo_variability = 0
        
        onset_density = len(onset_frames) / len(y) if len(y) > 0 else 0
        rms_variability = np.std(rms)
        contrast_variability = np.std(spectral_contrast)
        
        engineered_features = np.array([
            tempo, tempo_variability, hp_ratio, onset_density,
            rms_variability, contrast_variability,
            np.mean(rms), np.std(rms),
            np.mean(spectral_contrast), np.std(spectral_contrast)
        ])
        
        # Combine all features
        all_features = np.concatenate([
            baseline_features,
            emotion_features,
            engineered_features
        ])
        
        return all_features
        
    except Exception as e:
        print(f" Error extracting features from audio array: {e}")
        return None

def create_enhanced_classifier():
    """Create enhanced classifier with hyperparameter tuning"""
    
    print(" Creating Enhanced Classifier...")
    
    # Random Forest with class weights
    rf_classifier = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Gradient Boosting for better emotion detection
    gb_classifier = GradientBoostingClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
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

def hyperparameter_tuning(X, y):
    """Perform hyperparameter tuning"""
    
    print(" Performing Hyperparameter Tuning...")
    
    # Parameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create classifier for tuning
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Perform grid search with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f" Best parameters: {grid_search.best_params_}")
    print(f" Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_enhanced_model():
    """Train enhanced model with all improvements"""
    
    print(" Smart Playlist Matcher - ENHANCED Training")
    print("=" * 60)
    
    # Initialize classifier
    classifier = SmartPlaylistClassifier()
    
    # Dataset paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    audio_dir = "datasets/deam/DEAM/audio"
    valence_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
    arousal_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/dynamic_annotations_averaged_songs_1-2000/dynamic_annotations_averaged_songs_1_2000.csv"
    
    # Load original data with enhanced features
    print(" Loading data with enhanced features...")
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
    for song_id in range(2, 2002):  # DEAM dataset song IDs
        mp3_file = audio_path / f"{song_id}.mp3"
        
        if not mp3_file.exists():
            continue
        
        try:
            # Extract enhanced features
            features = extract_emotion_enhanced_features(mp3_file)
            
            if features is not None:
                # Get mood from annotations
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
                    
        except Exception as e:
            print(f" Error processing {song_id}: {e}")
            continue
    
    if len(X_original) == 0:
        print(" No training data available!")
        return
    
    print(f" Original samples: {len(X_original)}")
    print(f" Feature dimension: {len(X_original[0])}")
    
    # Convert to numpy arrays
    X = np.array(X_original)
    y = np.array(y_original)
    
    # Scale features
    print(" Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hyperparameter tuning
    best_model = hyperparameter_tuning(X_scaled, y)
    
    # Cross-validation evaluation
    print(" Performing Cross-Validation...")
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='f1_weighted')
    print(f" Cross-validation F1 scores: {cv_scores}")
    print(f" Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    print(" Training Final Enhanced Model...")
    best_model.fit(X_scaled, y)
    
    # Detailed evaluation
    print(" Detailed Model Evaluation...")
    y_pred = best_model.predict(X_scaled)
    
    print("\n ENHANCED CLASSIFICATION REPORT:")
    print("=" * 50)
    print(classification_report(y, y_pred, target_names=MOODS))
    
    print("\n CONFUSION MATRIX:")
    print("=" * 50)
    cm = confusion_matrix(y, y_pred)
    print("Predicted →")
    print("Actual ↓")
    for i, mood in enumerate(MOODS):
        row_str = f"{mood:8} |"
        for j in range(len(MOODS)):
            row_str += f" {cm[i][j]:4}"
        print(row_str)
    
    # Save enhanced model
    model_dir = ROOT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'moods': MOODS,
        'is_trained': True,
        'feature_count': len(X_original[0])
    }
    
    model_path = model_dir / "smart_playlist_classifier_enhanced.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n Enhanced model saved to: {model_path}")
    print(f" Feature count: {len(X_original[0])}")
    
    # Test prediction
    print(f"\n Testing enhanced prediction...")
    test_audio = ROOT_DIR / audio_dir / "1.mp3"
    if test_audio.exists():
        try:
            features = extract_emotion_enhanced_features(test_audio)
            if features is not None:
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction = best_model.predict(features_scaled)[0]
                probabilities = best_model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
                
                print(f"🎵 Enhanced prediction:")
                print(f"  Mood: {prediction} (confidence: {confidence:.2f})")
                print(f"  Probabilities: {dict(zip(MOODS, probabilities))}")
        except Exception as e:
            print(f" Test prediction failed: {e}")
    
    print(f"\n Enhanced Training Complete!")

if __name__ == "__main__":
    train_enhanced_model()
