# Emotion-Classification

## Executive Summary
This notebook implements and compares two Deep Learning models, **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)**, for classifying emotions from time-series data. The analysis utilizes a dataset of 2,132 samples with 2,549 features each.

**Key Findings:**
- **LSTM Model** achieved superior performance with a Test Accuracy of **96.56%**.
- **GRU Model** achieved strong performance but slightly lower, with a Test Accuracy of **94.06%**.
- Both models converged well, showing stable training dynamics with Early Stopping preventing overfitting.

##  Methodology

### 1- Preprocessing
A robust preprocessing pipeline was implemented:
1.  **Scaling**: Features scaled using `StandardScaler` to normalize distributions.
2.  **Encoding**: Target labels encoded with `LabelEncoder` and converted to categorical one-hot vectors.
3.  **Splitting**: Data partitioned into Train, Validation, and Test sets (Train: ~70%, Val: ~15%, Test: ~15%).
4.  **Reshaping**: High-dimensional feature space reshaped into temporal sequences for RNN input.
    -   **Timesteps**: 32
    -   **Features per Timestep**: 79 (Derived from total features 2549 ≈ 32 * 79).

### 2- Model Architectures
Both models share a similar 3-layer recurrent architecture designed for sequential data processing:

**LSTM Architecture:**
-   Input Layer
-   **LSTM Layer 1**: 128 units (return sequences) + Dropout (0.3)
-   **LSTM Layer 2**: 64 units (return sequences) + Dropout (0.3)
-   **LSTM Layer 3**: 32 units + Dropout (0.3)
-   **Dense Layer**: 64 units (ReLU activation) + Dropout (0.3)
-   **Output Layer**: 3 units (Softmax activation)

**GRU Architecture:**
-   Mirrors the LSTM structure but replaces LSTM layers with GRU layers of identical unit counts (128 → 64 → 32).

### 3- Training Configuration
-   **Optimizer**: Adam (Learning rate: 0.001)
-   **Loss Function**: Categorical Crossentropy
-   **Callbacks**:
    -   `EarlyStopping`: Monitors validation loss (Patience = 5) to stop training when improvement stalls.
    -   `ReduceLROnPlateau`: Reduces learning rate by factor of 0.5 if validation loss plateaus.

## Results & Evaluation

### 1- Performance Comparison

| Metric | LSTM Model | GRU Model |
| :--- | :--- | :--- |
| **Test Loss** | **0.1258** | 0.1943 |
| **Test Accuracy** | **96.56%** | 94.06% |
| **Best Epoch** | Epoch 9 | Epoch 11 |

### 2- Analysis
-   **Convergence**: Both models converged within 20 epochs. The LSTM reached its optimal state slightly faster (Epoch 9) compared to the GRU (Epoch 11).
-   **Stability**: Loss curves indicate stable learning with no major signs of divergent overfitting, thanks to the dropout layers and callbacks.
-   **Confusion Matrix**: Both models show high diagonal density, indicating accurate classification across all three emotion classes.

## 5. Conclusion
Both recurrent neural network architectures proved highly effective for this task. However, the **LSTM model is the recommended choice**, offering a ~2.5% improvement in accuracy and a significantly lower loss compared to the GRU model. The ability of the LSTM to manage long-term dependencies appears to provide a distinctive edge for this specific feature set.
