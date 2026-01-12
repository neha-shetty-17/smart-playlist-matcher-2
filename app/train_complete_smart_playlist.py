import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import librosa
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Define mood mappings based on valence and arousal
def valence_arousal_to_mood(valence, arousal):
    """Convert valence and arousal values to discrete mood labels"""
    # Thresholds (1-9 scale)
    valence_threshold = 5.0  # High/Low valence
    arousal_threshold = 5.0  # High/Low arousal
    
    if valence >= valence_threshold and arousal >= arousal_threshold:
        return 'happy'  # High valence, high arousal
    elif valence >= valence_threshold and arousal < arousal_threshold:
        return 'calm'   # High valence, low arousal
    elif valence < valence_threshold and arousal >= arousal_threshold:
        return 'angry'  # Low valence, high arousal
    else:
        return 'sad'    # Low valence, low arousal

MOODS = ['happy', 'calm', 'angry', 'sad']

def augment_audio(y, sr, augmentation_type):
    """Apply specific augmentation to audio array"""
    if augmentation_type == "time_stretch_slow":
        return librosa.effects.time_stretch(y, rate=0.9), sr
    
    elif augmentation_type == "time_stretch_fast":
        return librosa.effects.time_stretch(y, rate=1.1), sr
    
    elif augmentation_type == "pitch_up":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=1), sr
    
    elif augmentation_type == "pitch_down":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-1), sr
    
    elif augmentation_type == "add_noise":
        noise = np.random.randn(len(y)) * 0.003
        return y + noise, sr
    
    elif augmentation_type == "volume_quiet":
        return y * 0.8, sr
    
    elif augmentation_type == "volume_loud":
        return y * 1.2, sr
    
    return y, sr

def extract_features_with_engineered_array(y, sr):
    """Extract features from audio array (not file path)"""
    try:
        # === BASELINE AUDIO FEATURES ===
        
        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # Energy/RMS
        rms = librosa.feature.rms(y=y)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # === ENGINEERED FEATURE 1: TEMPO VARIABILITY ===
        
        # Extract tempo across multiple windows to measure variability
        hop_length = 512
        frame_length = 2048
        
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # Track tempo over time using autocorrelation
        tempos = []
        window_size = sr * 4  # 4-second windows
        
        for i in range(0, len(y) - window_size, window_size // 2):
            window = y[i:i + window_size]
            if len(window) == window_size:
                try:
                    tempo, _ = librosa.beat.beat_track(y=window, sr=sr, hop_length=hop_length)
                    tempos.append(tempo)
                except:
                    continue
        
        if len(tempos) > 1:
            tempo_mean = np.mean(tempos)
            tempo_std = np.std(tempos)  # TEMPO VARIABILITY - Key engineered feature
            tempo_cv = tempo_std / tempo_mean if tempo_mean > 0 else 0  # Coefficient of variation
        else:
            # Fallback to single tempo estimation
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            tempo_mean = tempo
            tempo_std = 0
            tempo_cv = 0
        
        # === ENGINEERED FEATURE 2: LOUDNESS VARIABILITY ===
        
        # Track loudness (RMS energy) over time
        loudness_values = []
        loudness_window_size = sr * 2  # 2-second windows
        
        for i in range(0, len(y) - loudness_window_size, loudness_window_size // 2):
            window = y[i:i + loudness_window_size]
            if len(window) == loudness_window_size:
                try:
                    rms_window = librosa.feature.rms(y=window)
                    loudness_values.append(np.mean(rms_window))
                except:
                    continue
        
        if len(loudness_values) > 1:
            loudness_mean = np.mean(loudness_values)
            loudness_std = np.std(loudness_values)  # LOUDNESS VARIABILITY - Key engineered feature
            loudness_cv = loudness_std / loudness_mean if loudness_mean > 0 else 0
        else:
            loudness_mean = np.mean(rms)
            loudness_std = np.std(rms)
            loudness_cv = loudness_std / loudness_mean if loudness_mean > 0 else 0
        
        # === ADDITIONAL ENGINEERED FEATURES ===
        
        # Dynamic range (max/min amplitude ratio)
        dynamic_range = np.max(np.abs(y)) / (np.min(np.abs(y)) + 1e-8)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        
        # Tonnetz (harmonic content)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        
        # Combine baseline features
        baseline_features = np.concatenate([
            mfccs_mean,                    # 13 features
            mfccs_std,                     # 13 features  
            [np.mean(spectral_centroids)], # 1 feature
            [np.mean(spectral_rolloff)],   # 1 feature
            [np.mean(spectral_bandwidth)], # 1 feature
            [np.mean(zero_crossing_rate)], # 1 feature
            [np.mean(rms)],                # 1 feature
            chroma_mean,                    # 12 features
            spectral_contrast_mean,        # 7 features
            tonnetz_mean,                  # 6 features
            [dynamic_range]                # 1 feature
        ])
        
        # Add engineered features
        engineered_features = np.array([
            tempo_mean,        # Overall tempo
            tempo_std,         # TEMPO VARIABILITY (rhythmic stability)
            tempo_cv,          # Normalized tempo variability
            loudness_mean,     # Overall loudness
            loudness_std,      # LOUDNESS VARIABILITY (energy dynamics)
            loudness_cv        # Normalized loudness variability
        ])
        
        # Combine all features
        features = np.concatenate([baseline_features, engineered_features])
        
        return features
        
    except Exception as e:
        print(f"❌ Error extracting features from array: {e}")
        return None

def extract_features_with_engineered(audio_path, duration=30):
    """Extract audio features including engineered tempo and loudness variability"""
    try:
        print(f"🎵 Processing: {audio_path}")
        
        # Load audio file
        y, sr = librosa.load(audio_path, duration=duration)
        
        # === BASELINE AUDIO FEATURES ===
        
        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # Energy/RMS
        rms = librosa.feature.rms(y=y)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # === ENGINEERED FEATURE 1: TEMPO VARIABILITY ===
        print("  📊 Extracting tempo variability...")
        
        # Extract tempo across multiple windows to measure variability
        hop_length = 512
        frame_length = 2048
        
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # Track tempo over time using autocorrelation
        tempos = []
        window_size = sr * 4  # 4-second windows
        
        for i in range(0, len(y) - window_size, window_size // 2):
            window = y[i:i + window_size]
            if len(window) == window_size:
                try:
                    tempo, _ = librosa.beat.beat_track(y=window, sr=sr, hop_length=hop_length)
                    tempos.append(tempo)
                except:
                    continue
        
        if len(tempos) > 1:
            tempo_mean = np.mean(tempos)
            tempo_std = np.std(tempos)  # TEMPO VARIABILITY - Key engineered feature
            tempo_cv = tempo_std / tempo_mean if tempo_mean > 0 else 0  # Coefficient of variation
            print(f"    Mean BPM: {tempo_mean:.1f}, Variability: {tempo_std:.2f}")
        else:
            # Fallback to single tempo estimation
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_mean = tempo
            tempo_std = 0
            tempo_cv = 0
            print(f"    Single tempo: {tempo_mean:.1f} BPM")
        
        # === ENGINEERED FEATURE 2: LOUDNESS VARIABILITY ===
        print("  🔊 Extracting loudness variability...")
        
        # Track loudness (RMS energy) over time
        loudness_values = []
        loudness_window_size = sr * 2  # 2-second windows
        
        for i in range(0, len(y) - loudness_window_size, loudness_window_size // 2):
            window = y[i:i + loudness_window_size]
            if len(window) == loudness_window_size:
                try:
                    rms_window = librosa.feature.rms(y=window)
                    loudness_values.append(np.mean(rms_window))
                except:
                    continue
        
        if len(loudness_values) > 1:
            loudness_mean = np.mean(loudness_values)
            loudness_std = np.std(loudness_values)  # LOUDNESS VARIABILITY - Key engineered feature
            loudness_cv = loudness_std / loudness_mean if loudness_mean > 0 else 0
            print(f"    Mean Loudness: {loudness_mean:.4f}, Variability: {loudness_std:.4f}")
        else:
            loudness_mean = np.mean(rms)
            loudness_std = np.std(rms)
            loudness_cv = loudness_std / loudness_mean if loudness_mean > 0 else 0
            print(f"    Single loudness: {loudness_mean:.4f}")
        
        # === ADDITIONAL ENGINEERED FEATURES ===
        
        # Dynamic range (max/min amplitude ratio)
        dynamic_range = np.max(np.abs(y)) / (np.min(np.abs(y)) + 1e-8)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        
        # Tonnetz (harmonic content)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        
        # Combine baseline features
        baseline_features = np.concatenate([
            mfccs_mean,                    # 13 features
            mfccs_std,                     # 13 features  
            [np.mean(spectral_centroids)], # 1 feature
            [np.mean(spectral_rolloff)],   # 1 feature
            [np.mean(spectral_bandwidth)], # 1 feature
            [np.mean(zero_crossing_rate)], # 1 feature
            [np.mean(rms)],                # 1 feature
            chroma_mean,                    # 12 features
            spectral_contrast_mean,        # 7 features
            tonnetz_mean,                  # 6 features
            [dynamic_range]                # 1 feature
        ])
        
        # Add engineered features
        engineered_features = np.array([
            tempo_mean,        # Overall tempo
            tempo_std,         # TEMPO VARIABILITY (rhythmic stability)
            tempo_cv,          # Normalized tempo variability
            loudness_mean,     # Overall loudness
            loudness_std,      # LOUDNESS VARIABILITY (energy dynamics)
            loudness_cv        # Normalized loudness variability
        ])
        
        # Combine all features
        features = np.concatenate([baseline_features, engineered_features])
        
        print(f"  ✅ Extracted {len(features)} features")
        return features
        
    except Exception as e:
        print(f"❌ Error extracting features from {audio_path}: {e}")
        return None

class SmartPlaylistClassifier:
    """Random Forest classifier for mood-based playlist generation with engineered features"""
    
    def __init__(self):
        self.moods = MOODS
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def prepare_training_data(self, audio_dir, valence_file, arousal_file):
        """Prepare training data from DEAM dataset"""
        ROOT_DIR = Path(__file__).resolve().parent.parent
        
        print("🎯 Loading DEAM dataset annotations...")
        
        # Load annotations
        valence_df = pd.read_csv(ROOT_DIR / valence_file)
        arousal_df = pd.read_csv(ROOT_DIR / arousal_file)
        
        # Clean column names (remove leading spaces)
        valence_df.columns = valence_df.columns.str.strip()
        arousal_df.columns = arousal_df.columns.str.strip()
        
        # Merge annotations
        annotations = pd.merge(valence_df, arousal_df, on='song_id')
        
        # Convert to mood labels
        annotations['mood'] = annotations.apply(
            lambda row: valence_arousal_to_mood(row['valence_mean'], row['arousal_mean']), 
            axis=1
        )
        
        print(f"📊 Loaded {len(annotations)} annotated songs")
        print("🎵 Mood distribution:")
        print(annotations['mood'].value_counts())
        
        # Extract features for each song
        X = []
        y = []
        missing_files = []
        
        print("\n🔧 Extracting features with engineered tempo & loudness variability...")
        
        for idx, row in annotations.iterrows():
            song_id = row['song_id']
            audio_path = ROOT_DIR / audio_dir / f"{song_id}.mp3"
            
            if not audio_path.exists():
                missing_files.append(song_id)
                continue
            
            features = extract_features_with_engineered(str(audio_path))
            if features is not None:
                X.append(features)
                y.append(row['mood'])
            else:
                missing_files.append(song_id)
                continue
        
        if missing_files:
            print(f"⚠️  Missing audio files: {len(missing_files)}")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Successfully processed {len(X)} songs")
        return X, y
    
    def train(self, X, y):
        """Train the Random Forest classifier"""
        print("\n🚀 Training Smart Playlist Classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Scale features
        print("🔧 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("🎯 Training Random Forest model...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n🎉 Model Performance:")
        print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.moods))
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        print(f"\n🎯 Top 15 Most Important Features:")
        top_indices = np.argsort(feature_importance)[-15:][::-1]
        
        feature_names = [
            # MFCC features
            'MFCC1_mean', 'MFCC2_mean', 'MFCC3_mean', 'MFCC4_mean', 'MFCC5_mean',
            'MFCC6_mean', 'MFCC7_mean', 'MFCC8_mean', 'MFCC9_mean', 'MFCC10_mean',
            'MFCC11_mean', 'MFCC12_mean', 'MFCC13_mean',
            'MFCC1_std', 'MFCC2_std', 'MFCC3_std', 'MFCC4_std', 'MFCC5_std',
            'MFCC6_std', 'MFCC7_std', 'MFCC8_std', 'MFCC9_std', 'MFCC10_std',
            'MFCC11_std', 'MFCC12_std', 'MFCC13_std',
            # Spectral features
            'Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Bandwidth',
            'Zero_Crossing_Rate', 'RMS_Energy',
            # Chroma features
            'Chroma1', 'Chroma2', 'Chroma3', 'Chroma4', 'Chroma5',
            'Chroma6', 'Chroma7', 'Chroma8', 'Chroma9', 'Chroma10',
            'Chroma11', 'Chroma12',
            # Spectral contrast
            'Contrast1', 'Contrast2', 'Contrast3', 'Contrast4', 'Contrast5',
            'Contrast6', 'Contrast7',
            # Tonnetz
            'Tonnetz1', 'Tonnetz2', 'Tonnetz3', 'Tonnetz4', 'Tonnetz5', 'Tonnetz6',
            'Dynamic_Range',
            # Engineered features
            'Tempo_Mean', 'Tempo_Variability', 'Tempo_CV',
            'Loudness_Mean', 'Loudness_Variability', 'Loudness_CV'
        ]
        
        for i, idx in enumerate(top_indices):
            feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            importance = feature_importance[idx]
            print(f"{i+1:2d}. {feature_name:25s}: {importance:.4f}")
        
        # Highlight engineered features
        engineered_indices = [len(feature_names)-6, len(feature_names)-5, len(feature_names)-4,
                            len(feature_names)-3, len(feature_names)-2, len(feature_names)-1]
        print(f"\n🔧 Engineered Features Performance:")
        for idx in engineered_indices:
            if idx < len(feature_importance):
                feature_name = feature_names[idx]
                importance = feature_importance[idx]
                print(f"  {feature_name:25s}: {importance:.4f}")
        
        return accuracy, cm
    
    def predict_mood(self, audio_path):
        """Predict mood for a single audio file and return tempo information"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = extract_features_with_engineered(audio_path)
        if features is None:
            raise ValueError(f"Could not extract features from {audio_path}")
        
        # Extract engineered features for return
        feature_names = [
            # ... (same as in training)
            'Tempo_Mean', 'Tempo_Variability', 'Tempo_CV',
            'Loudness_Mean', 'Loudness_Variability', 'Loudness_CV'
        ]
        
        # Engineered features are at the end
        tempo_mean = features[-6]  # Overall tempo
        tempo_std = features[-5]   # TEMPO VARIABILITY  
        tempo_cv = features[-4]    # Coefficient of variation
        loudness_mean = features[-3]  # Overall loudness
        loudness_std = features[-2]   # LOUDNESS VARIABILITY
        loudness_cv = features[-1]    # Coefficient of variation
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        mood = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled))
        
        return {
            'mood': mood,
            'confidence': confidence,
            'tempo': {
                'mean_bpm': round(tempo_mean, 2),
                'variability': round(tempo_std, 2),
                'coefficient_of_variation': round(tempo_cv, 3)
            },
            'loudness': {
                'mean_energy': round(loudness_mean, 4),
                'variability': round(loudness_std, 4),
                'coefficient_of_variation': round(loudness_cv, 3)
            }
        }
    
    def generate_playlist(self, audio_directory, target_mood, max_songs=20):
        """Generate a playlist of songs with the specified mood"""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating playlists")
        
        ROOT_DIR = Path(__file__).resolve().parent.parent
        audio_dir = ROOT_DIR / audio_directory
        
        playlist = []
        
        print(f"🎵 Generating {target_mood} playlist from {audio_directory}...")
        
        for audio_file in audio_dir.glob("*.mp3"):
            if len(playlist) >= max_songs:
                break
                
            try:
                result = self.predict_mood(str(audio_file))
                if result['mood'] == target_mood:
                    playlist.append({
                        'filename': audio_file.name,
                        'mood': result['mood'],
                        'confidence': result['confidence'],
                        'tempo': result['tempo'],
                        'loudness': result['loudness']
                    })
                    print(f"  ✅ Added: {audio_file.name} (confidence: {result['confidence']:.2f})")
            except Exception as e:
                print(f"  ❌ Error processing {audio_file}: {e}")
                continue
        
        print(f"🎉 Generated playlist with {len(playlist)} songs")
        return playlist
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'moods': self.moods,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.moods = model_data['moods']
        self.is_trained = model_data['is_trained']
        
        print(f"📂 Model loaded from {filepath}")

def prepare_augmented_training_data(audio_dir, valence_file, arousal_file):
    """Prepare training data with augmentation"""
    
    # Initialize classifier
    classifier = SmartPlaylistClassifier()
    
    # Load original data using classifier method
    X_original, y_original = classifier.prepare_training_data(audio_dir, valence_file, arousal_file)
    
    # Initialize augmented datasets
    X_augmented = [X_original]
    y_augmented = [y_original]
    
    # Define augmentation types
    augmentations = [
        "time_stretch_slow",      # 10% slower
        "time_stretch_fast",      # 10% faster
        "pitch_up",              # Up 1 semitone
        "pitch_down",            # Down 1 semitone
        "add_noise",             # Light noise
        "volume_quiet",          # 80% volume
        "volume_loud"           # 120% volume
    ]
    
    print(f"🔄 Starting data augmentation...")
    print(f"📊 Original samples: {len(X_original)}")
    
    # Process each original sample
    for i, (features, mood) in enumerate(zip(X_original, y_original)):
        if i % 100 == 0:
            print(f"  Processing sample {i}/{len(X_original)}")
        
        # Get original audio file path
        song_id = i + 2  # Adjust for song_id mapping
        audio_path = Path(audio_dir) / f"{song_id}.mp3"
        
        if not audio_path.exists():
            continue
            
        # Load original audio once
        try:
            y_orig, sr_orig = librosa.load(str(audio_path), duration=30)
        except Exception as e:
            print(f"❌ Could not load {audio_path}: {e}")
            continue
        
        # Apply augmentations
        for aug_type in augmentations:
            try:
                # Apply augmentation
                y_aug, sr_aug = augment_audio(y_orig, sr_orig, aug_type)
                
                # Extract features from augmented audio
                aug_features = extract_features_with_engineered_array(y_aug, sr_aug)
                
                if aug_features is not None:
                    X_augmented.append(aug_features)
                    y_augmented.append(mood)
                    
            except Exception as e:
                print(f"❌ Augmentation failed for {song_id} - {aug_type}: {e}")
                continue
    
    print(f"✅ Total augmented samples: {len(X_augmented)}")
    return np.array(X_augmented), np.array(y_augmented)

def main():
    """Main training function with augmentation"""
    print("🎵 Smart Playlist Matcher - AUGMENTED Training")
    print("=" * 60)
    
    # Initialize classifier
    classifier = SmartPlaylistClassifier()
    
    # Dataset paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    audio_dir = "datasets/deam/DEAM/audio"
    valence_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
    arousal_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/dynamic_annotations_averaged_songs_1-2000/dynamic_annotations_averaged_songs_1_2000.csv"
    
    # Prepare AUGMENTED training data
    print("🔄 Starting data augmentation...")
    X_original, y_original = classifier.prepare_training_data(audio_dir, valence_file, arousal_file)
    
    if len(X_original) == 0:
        print("❌ No training data available!")
        return
    
    print(f"📊 Original samples: {len(X_original)}")
    
    # Initialize augmented datasets
    X_augmented = []
    y_augmented = []
    
    # Add original data first
    for features, mood in zip(X_original, y_original):
        X_augmented.append(features)
        y_augmented.append(mood)
    
    # Define augmentation types
    augmentations = [
        "time_stretch_slow",      # 10% slower
        "time_stretch_fast",      # 10% faster
        "pitch_up",              # Up 1 semitone
        "pitch_down",            # Down 1 semitone
        "add_noise",             # Light noise
        "volume_quiet",          # 80% volume
        "volume_loud"           # 120% volume
    ]
    
    # Process each original sample with limited augmentations
    for i, (features, mood) in enumerate(zip(X_original, y_original)):
        if i % 50 == 0:
            print(f"  Processing sample {i}/{len(X_original)}")
        
        # Get original audio file path
        song_id = i + 2  # Adjust for song_id mapping
        audio_path = Path(audio_dir) / f"{song_id}.mp3"
        
        if not audio_path.exists():
            continue
            
        # Load original audio once
        try:
            y_orig, sr_orig = librosa.load(str(audio_path), duration=30)
        except Exception as e:
            print(f"❌ Could not load {audio_path}: {e}")
            continue
        
        # Apply limited augmentations
        for aug_type in augmentations:
            try:
                # Apply augmentation
                y_aug, sr_aug = augment_audio(y_orig, sr_orig, aug_type)
                
                # Extract features from augmented audio
                aug_features = extract_features_with_engineered_array(y_aug, sr_aug)
                
                if aug_features is not None:
                    # Ensure 1D array for sklearn
                    if aug_features.ndim > 1:
                        aug_features = aug_features.flatten()
                    X_augmented.append(aug_features)
                    y_augmented.append(mood)
                    
            except Exception as e:
                print(f"❌ Augmentation failed for {song_id} - {aug_type}: {e}")
                continue
    
    print(f"✅ Total augmented samples: {len(X_augmented)}")
    
    # Convert to numpy arrays and ensure 2D shape
    X = np.array(X_augmented)
    y = np.array(y_augmented)
    
    # Ensure X is 2D - handle nested arrays from augmentation
    if X.ndim == 3:
        # Flatten first two dimensions if we have (n_samples, 1, n_features)
        X = X.reshape(X.shape[0], -1)
    elif X.ndim > 2:
        # Flatten all but first dimension
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 1:
        X = X.reshape(-1, 1)
    
    print(f"\n📊 Training with {len(X)} samples (original + augmented)")
    print(f"📏 Feature array shape: {X.shape}")
    
    # Split data with proper test size
    if len(X) < 20:
        print("⚠️  Too few samples for splitting, using all for training")
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # Scale features
    print("🎯 Scaling features...")
    classifier.scaler = StandardScaler()
    X_train_scaled = classifier.scaler.fit_transform(X_train)
    X_test_scaled = classifier.scaler.transform(X_test)
    
    # Train model
    print("🎯 Training Random Forest...")
    classifier.model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    classifier.model.fit(X_train_scaled, y_train)
    classifier.is_trained = True
    classifier.moods = ['happy', 'calm', 'angry', 'sad']
    
    # Evaluate
    print("📊 Evaluating model...")
    y_pred = classifier.model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎉 Augmented Training Complete!")
    print(f"📊 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save augmented model
    model_dir = ROOT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_data = {
        'model': classifier.model,
        'scaler': classifier.scaler,
        'moods': classifier.moods,
        'is_trained': classifier.is_trained
    }
    
    model_path = model_dir / "smart_playlist_classifier_augmented.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Augmented model saved to: {model_path}")
    
    # Test prediction
    print(f"\n🧪 Testing prediction on sample audio...")
    test_audio = ROOT_DIR / audio_dir / "1.mp3"
    if test_audio.exists():
        try:
            result = classifier.predict_mood(str(test_audio))
            print(f"🎵 Sample prediction:")
            print(f"  Mood: {result['mood']} (confidence: {result['confidence']:.2f})")
            print(f"  Tempo: {result['tempo']['mean_bpm']} BPM (variability: {result['tempo']['variability']})")
            print(f"  Loudness: {result['loudness']['mean_energy']:.4f} (variability: {result['loudness']['variability']:.4f})")
        except Exception as e:
            print(f"❌ Test prediction failed: {e}")

def main_original():
    """Original main training function (for comparison)"""
    print("🎵 Smart Playlist Matcher - Original Training (No Augmentation)")
    print("=" * 60)
    
    # Initialize classifier
    classifier = SmartPlaylistClassifier()
    
    # Dataset paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    audio_dir = "datasets/deam/DEAM/audio"
    valence_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
    arousal_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/dynamic_annotations_averaged_songs_1-2000/dynamic_annotations_averaged_songs_1_2000.csv"
    
    # Prepare training data
    X, y = classifier.prepare_training_data(audio_dir, valence_file, arousal_file)
    
    if len(X) == 0:
        print("❌ No training data available!")
        return
    
    # Train model
    accuracy, cm = classifier.train(X, y)
    
    # Save model
    model_dir = ROOT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    classifier.save_model(model_dir / "smart_playlist_classifier_original.pkl")
    
    print(f"\n🎉 Original Training Complete!")
    print(f"📊 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"💾 Original model saved to: models/smart_playlist_classifier_original.pkl")

if __name__ == "__main__":
    main()
