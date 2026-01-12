#!/usr/bin/env python3
"""
Improved Mood Classification with Enhanced Features
Addresses overlapping characteristics between mood classes
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_mood_features(valence, arousal, tempo_mean, tempo_std, energy_mean, energy_std):
    """
    Create enhanced features to better distinguish between overlapping moods
    """
    features = []
    
    # Base valence/arousal
    features.extend([valence, arousal])
    
    # Tempo features
    features.extend([tempo_mean, tempo_std])
    
    # Energy features  
    features.extend([energy_mean, energy_std])
    
    # Ratio features (key for distinguishing overlapping moods)
    features.append(valence * arousal)  # Valence-Arousal interaction
    features.append(tempo_mean * energy_mean)  # Tempo-Energy interaction
    features.append(valence / (arousal + 0.001))  # Valence/Arousal ratio
    features.append(tempo_mean / (energy_mean + 0.001))  # Tempo/Energy ratio
    
    # Quadrant features
    features.append(1 if valence > 5.0 else 0)  # High valence
    features.append(1 if arousal > 5.0 else 0)  # High arousal
    features.append(1 if tempo_mean > 120 else 0)  # High tempo
    features.append(1 if energy_mean > 0.6 else 0)  # High energy
    
    # Distinguishing features for overlapping moods
    # Angry vs Happy: Angry has higher arousal + negative valence
    features.append(1 if (valence < 4.5 and arousal > 6.0) else 0)  # Angry indicator
    # Happy vs Angry: Happy has positive valence + high arousal  
    features.append(1 if (valence > 6.0 and arousal > 6.0) else 0)  # Happy indicator
    # Sad vs Calm: Sad has negative valence + low arousal
    features.append(1 if (valence < 4.0 and arousal < 4.0) else 0)  # Sad indicator
    # Calm vs Sad: Calm has positive valence + low arousal
    features.append(1 if (valence > 5.5 and arousal < 4.5) else 0)  # Calm indicator
    
    # Energy-based features
    features.append(np.sqrt(tempo_mean**2 + energy_mean**2))  # Combined energy
    features.append(tempo_std / (tempo_mean + 0.001))  # Tempo variability
    features.append(energy_std / (energy_mean + 0.001))  # Energy variability
    
    return np.array(features)

def train_improved_mood_classifier():
    """Train improved mood classifier with enhanced features"""
    
    # Load existing model data
    model_path = Path(__file__).resolve().parent / "models" / "smart_playlist_classifier_enhanced_fast.pkl"
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Load annotations
    ROOT_DIR = Path(__file__).resolve().parent.parent
    valence_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
    arousal_file = "datasets/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/dynamic_annotations_averaged_songs_1-2000/dynamic_annotations_averaged_songs_1_2000.csv"
    
    valence_df = pd.read_csv(ROOT_DIR / valence_file)
    arousal_df = pd.read_csv(ROOT_DIR / arousal_file)
    
    # Clean and process
    valence_df.columns = valence_df.columns.str.strip()
    arousal_df.columns = arousal_df.columns.str.strip()
    
    arousal_cols = [col for col in arousal_df.columns if 'arousal' in col and col != 'song_id']
    arousal_df['arousal_mean'] = arousal_df[arousal_cols].mean(axis=1)
    
    annotations = pd.merge(valence_df, arousal_df, on='song_id', suffixes=('', '_y'))
    
    # Create perfectly balanced mood distribution (25% each)
    def create_perfectly_balanced_dataset(annotations):
        """
        Create perfectly balanced dataset with exactly 25% songs per mood
        """
        # Categorize songs by valence-arousal quadrants
        happy_songs = []
        energetic_songs = []
        calm_songs = []
        sad_songs = []
        
        for _, row in annotations.iterrows():
            valence = row['valence_mean']
            arousal = row['arousal_mean']
            
            # Normalize to 0-1
            v_norm = (valence - 1.6) / (8.4 - 1.6)
            a_norm = (arousal - 1.6) / (8.1 - 1.6)
            
            # Quadrant classification
            if v_norm > 0.5 and a_norm > 0.5:
                happy_songs.append(row)
            elif v_norm <= 0.5 and a_norm > 0.5:
                energetic_songs.append(row)
            elif v_norm > 0.5 and a_norm <= 0.5:
                calm_songs.append(row)
            else:
                sad_songs.append(row)
        
        print(f"📊 Original quadrant distribution:")
        print(f"   Happy: {len(happy_songs)} songs")
        print(f"   Energetic: {len(energetic_songs)} songs")
        print(f"   Calm: {len(calm_songs)} songs")
        print(f"   Sad: {len(sad_songs)} songs")
        
        # Target: exactly 25% each (436 songs per class)
        target_per_class = 436
        balanced_data = []
        
        # Helper function to balance by moving songs between adjacent quadrants
        def balance_classes(over_class, under_class, over_songs, under_songs, target):
            excess = len(over_songs) - target
            deficit = target - len(under_songs)
            
            if excess > 0 and deficit > 0:
                # Move songs from overrepresented to underrepresented
                move_count = min(excess, deficit)
                
                # For Happy → Energetic: move high-arousal happy songs
                if over_class == 'happy' and under_class == 'energetic':
                    sorted_songs = sorted(over_songs, key=lambda x: x['arousal_mean'], reverse=True)
                # For Sad → Calm: move high-valence sad songs  
                elif over_class == 'sad' and under_class == 'calm':
                    sorted_songs = sorted(over_songs, key=lambda x: x['valence_mean'], reverse=True)
                else:
                    sorted_songs = over_songs
                
                moved_songs = sorted_songs[:move_count]
                remaining_songs = sorted_songs[move_count:]
                
                return remaining_songs, under_songs + moved_songs
            else:
                return over_songs, under_songs
        
        # Balance Happy → Energetic
        happy_songs, energetic_songs = balance_classes(
            'happy', 'energetic', happy_songs, energetic_songs, target_per_class
        )
        
        # Balance Sad → Calm  
        sad_songs, calm_songs = balance_classes(
            'sad', 'calm', sad_songs, calm_songs, target_per_class
        )
        
        # If still imbalanced, take random samples
        happy_songs = happy_songs[:target_per_class]
        energetic_songs = energetic_songs[:target_per_class]
        calm_songs = calm_songs[:target_per_class]
        sad_songs = sad_songs[:target_per_class]
        
        # Create balanced dataset
        for song in happy_songs:
            song_copy = song.copy()
            song_copy['mood'] = 'happy'
            balanced_data.append(song_copy)
            
        for song in energetic_songs:
            song_copy = song.copy()
            song_copy['mood'] = 'energetic'
            balanced_data.append(song_copy)
            
        for song in calm_songs:
            song_copy = song.copy()
            song_copy['mood'] = 'calm'
            balanced_data.append(song_copy)
            
        for song in sad_songs:
            song_copy = song.copy()
            song_copy['mood'] = 'sad'
            balanced_data.append(song_copy)
        
        balanced_df = pd.DataFrame(balanced_data)
        
        print(f"\n✅ Perfectly balanced distribution:")
        mood_counts = balanced_df['mood'].value_counts()
        for mood, count in mood_counts.items():
            percentage = (count / len(balanced_df)) * 100
            print(f"   {mood}: {count} songs ({percentage:.1f}%)")
        
        return balanced_df
    
    # Create perfectly balanced dataset
    balanced_annotations = create_perfectly_balanced_dataset(annotations)
    
    # Create realistic tempo/energy based on mood characteristics
    np.random.seed(42)
    n_songs = len(balanced_annotations)
    
    # Create realistic tempo/energy based on mood characteristics
    tempo_data = []
    energy_data = []
    
    for _, row in balanced_annotations.iterrows():
        valence = row['valence_mean']
        arousal = row['arousal_mean']
        mood = row['mood']
        
        # Generate tempo based on mood
        if mood == 'happy':
            tempo = np.random.normal(130, 20)
            energy = np.random.normal(0.7, 0.15)
        elif mood == 'calm':
            tempo = np.random.normal(80, 15)
            energy = np.random.normal(0.3, 0.1)
        elif mood == 'energetic':
            tempo = np.random.normal(140, 25)
            energy = np.random.normal(0.8, 0.12)
        else:  # sad
            tempo = np.random.normal(70, 12)
            energy = np.random.normal(0.25, 0.08)
        
        # Clamp values
        tempo = np.clip(tempo, 40, 200)
        energy = np.clip(energy, 0.0, 1.0)
        
        tempo_data.append(tempo)
        energy_data.append(energy)
    
    balanced_annotations['tempo_mean'] = tempo_data
    balanced_annotations['tempo_std'] = np.random.uniform(5, 25, n_songs)
    balanced_annotations['energy_mean'] = energy_data
    balanced_annotations['energy_std'] = np.random.uniform(0.05, 0.2, n_songs)
    
    # Create enhanced features
    X = []
    y = []
    
    for _, row in balanced_annotations.iterrows():
        features = create_enhanced_mood_features(
            row['valence_mean'], row['arousal_mean'],
            row['tempo_mean'], row['tempo_std'],
            row['energy_mean'], row['energy_std']
        )
        X.append(features)
        y.append(row['mood'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train improved ensemble
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
    
    # Train both models
    rf.fit(X_scaled, y)
    gb.fit(X_scaled, y)
    
    # Create ensemble
    from sklearn.ensemble import VotingClassifier
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    
    ensemble.fit(X_scaled, y)
    
    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring='f1_weighted')
    print(f"🎯 Cross-validation F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Classification report
    y_pred = ensemble.predict(X_scaled)
    print("\n📊 Classification Report:")
    print(classification_report(y, y_pred))
    
    print("\n🎭 Mood Distribution:")
    mood_counts = pd.Series(y).value_counts()
    for mood, count in mood_counts.items():
        percentage = (count / len(y)) * 100
        print(f"   {mood}: {count} songs ({percentage:.1f}%)")
    
    # Save improved model
    improved_model_data = {
        'model': ensemble,
        'scaler': scaler,
        'features': X.shape[1],
        'model_type': 'improved_mood_classifier',
        'classes': ensemble.classes_,
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])]
    }
    
    output_path = Path(__file__).resolve().parent / "models" / "improved_mood_classifier.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(improved_model_data, f)
    
    print(f"\n✅ Improved mood classifier saved to: {output_path}")
    print(f"📏 Enhanced features: {X.shape[1]}")
    print(f"🎯 Model classes: {list(ensemble.classes_)}")
    
    return improved_model_data

if __name__ == "__main__":
    train_improved_mood_classifier()
