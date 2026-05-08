import numpy as np
import essentia
import essentia.standard as es
import os
from q3_solve import LogisticRegressionGD, OneVsRestClassifier, OneVsOneClassifier, SoftmaxClassifier, make_labels, make_labels_inverse

# Function to extract features. Uses essentia python library. Install it via pip. You may need to use an earlier version of python ( <= 3.12)
def extract_mfcc_wlpc_features(
    wav_path,
    frame_size=2048,
    hop_size=1024,
    sample_rate=44100,
    n_mfcc=15,
    n_mel_bands=40,
    lpc_order=15
):
    """
    Extract median MFCC and Warped LPC features from a WAV file using Essentia.

    Returns:
        mfcc_median: np.ndarray, shape (15,)
        wlpc_median: np.ndarray, shape (15,)
    """

    # ---- Load audio ----
    audio = es.MonoLoader(
        filename=wav_path,
        sampleRate=sample_rate
    )()

    # ---- MFCC chain ----
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    mfcc = es.MFCC(
        numberBands=n_mel_bands,
        numberCoefficients=n_mfcc
    )

    # ---- WLPC chain (implicit warping) ----
    warped_autocorr = es.WarpedAutoCorrelation(
        maxLag=lpc_order,
        sampleRate=sample_rate
    )

    lpc = es.LPC(order=lpc_order)

    mfcc_frames = []
    wlpc_frames = []

    # ---- Frame-wise processing ----
    for frame in es.FrameGenerator(
        audio,
        frameSize=frame_size,
        hopSize=hop_size,
        startFromZero=True
    ):
        # ---- MFCCs ----
        spec = spectrum(window(frame))
        _, mfcc_coeffs = mfcc(spec)
        mfcc_frames.append(mfcc_coeffs)

        # ---- WLPCs ----
        warped_r = warped_autocorr(frame)
        lpc_coeffs, _ = lpc(warped_r)
        wlpc_frames.append(lpc_coeffs)

    mfcc_frames = np.asarray(mfcc_frames)
    wlpc_frames = np.asarray(wlpc_frames)

    # ---- Median pooling ----
    mfcc_median = np.median(mfcc_frames, axis=0)
    wlpc_median = np.median(wlpc_frames, axis=0)

    return mfcc_median, wlpc_median




data_path = "./Dataset"
instruments = ['violin', 'clarinet', 'bassoon', 'saxphone']

# Construct the dataframe with features and labels
import pandas as pd
feature_list = []
labels = []
for instrument in instruments:
    instrument_path = os.path.join(data_path, instrument)
    for file in os.listdir(instrument_path):
        if file.endswith(".wav"):
            wav_path = os.path.join(instrument_path, file)
            mfcc_feat, wlpc_feat = extract_mfcc_wlpc_features(wav_path)
            # print(len(mfcc_feat), len(wlpc_feat))
            features = np.concatenate((mfcc_feat, wlpc_feat))
            feature_list.append(features)
            labels.append(instrument)



feature_names = [f"mfcc_{i+1}" for i in range(15)] + [f"wlpc_{i+1}" for i in range(16)]
df = pd.DataFrame(feature_list, columns=feature_names)
df['label'] = labels






print("Sanity Checks:")
print("-" * 50)
print("Function: make_labels and make_labels_inverse")
y = np.array(['violin', 'clarinet', 'bassoon', 'saxphone', 'violin', 'bassoon'])
y_encoded, dict_num_to_str = make_labels(y)
y_decoded = make_labels_inverse(y_encoded, dict_num_to_str)
print("Original labels:", y)
print("Encoded labels:", y_encoded)
print("Decoded labels:", y_decoded)

# Add simlar checks here for your functions to test them out.
print("-" * 50)


# Printing accuracies 
from sklearn.model_selection import train_test_split
df['label'], dict_num_to_str = make_labels(df['label'].values)
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['label']).values, df['label'].values, test_size=0.2, random_state=42)

# One-vs-Rest
ovr_clf = OneVsRestClassifier()
ovr_clf.fit(X_train, y_train)
ovr_pred = ovr_clf.predict(X_test)
ovr_acc = np.mean(ovr_pred == y_test)
print(f"One-vs-Rest Accuracy: {ovr_acc:.4f}")

# One-vs-One
ovo_clf = OneVsOneClassifier()
ovo_clf.fit(X_train, y_train)
ovo_pred = ovo_clf.predict(X_test)
ovo_acc = np.mean(ovo_pred == y_test)
print(f"One-vs-One Accuracy: {ovo_acc:.4f}")

# Softmax
softmax_clf = SoftmaxClassifier()
softmax_clf.fit(X_train, y_train)
softmax_pred = softmax_clf.predict(X_test)
softmax_acc = np.mean(softmax_pred == y_test)
print(f"Softmax Accuracy: {softmax_acc:.4f}")


import matplotlib.pyplot as plt
# Add plots for loss history.
# plt.plot(softmax_clf.softmax_loss_history, label='Softmax')
# plt.show()