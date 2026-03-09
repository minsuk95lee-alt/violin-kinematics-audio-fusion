# Decoding the Adjudicator’s Eye: Multimodal Sensory Integration in Musical Performance (Pilot Study)

This repository contains the pilot test pipeline ($N=14$) for my MSc Performance Science dissertation at the Royal College of Music. 

The objective is to computationally model human auditory and visual perception, specifically testing the **"Sight over Sound"** paradigm in elite musical adjudication by fusing visual kinematics and acoustic features.

## ⚙️ 1. Pipeline Architecture
Rather than relying on subjective observation, this study engineers an end-to-end multimodal machine learning pipeline:
* **Visual Kinematics (Computer Vision):** Extracted continuous time-series data (e.g., Kinematic Jerk, Spatial Occupancy) via `MediaPipe`, applying Butterworth and Savitzky-Golay filters to ensure signal integrity.
* **Acoustics (MIR):** Extracted concurrent spectral and temporal features (RMS Energy, Spectral Centroid, MFCCs) via `Librosa`.
* **Integration & Classification:** Benchmarked Visual-Only, Audio-Only, and Fused modalities using non-parametric statistics (Mann-Whitney U) and Linear SVM (LOOCV).

## 📊 2. Key Pilot Findings

### A. The Tension of Constraints: Kinematic Jerk
Statistical analysis revealed a strong trend connecting biomechanical constraints with movement efficiency. Under extreme technical constraints (Paganini), non-winners exhibited approximately **3x higher Kinematic Jerk** in the bow arm compared to winners, quantitatively capturing the concept of "effortless mastery."

![Boxplot Placeholder](link_to_your_pilot_test_results.png)

### B. Feature Importance: AI's Visual Preferences
A Random Forest baseline (Gini Importance) demonstrated that AI prioritizes *structural predictability* over mere quantity of movement. **Kinematic Entropy (29.2%)** and **Spatial Occupancy Std (24.5%)** were the strongest predictors of success, suggesting evaluators reward controlled, expansive physical framing.

![Feature Importance Placeholder](link_to_your_feature_importance.png)

### C. The Modality Question: Sight over Sound
A Linear SVM was trained to predict the outcome using distinct sensory modalities.
* **Visual-Only Model:** 64.3% Accuracy
* **Audio-Only Model:** 42.9% Accuracy
* **Fused Model:** 50.0% Accuracy

**Insight:** At the elite tier (where acoustic variance is minimal), visual kinematics dominate the decision-making process. The performance drop in the Fused Model computationally replicates the human cognitive phenomenon of sensory overload/interference described in Tsay (2013).

## 🚀 3. Next Steps (Macro-Dataset)
This pilot validates the mathematical robustness of the extraction pipeline. The next phase involves scaling the extraction to a global macro-dataset ($N=40$) to achieve robust statistical power and refine the multimodal early/late-fusion architectures.
