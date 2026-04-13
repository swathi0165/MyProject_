import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, TensorDataset

# =============================
# LOAD DATA (BALANCED SPEED)
# =============================
data = pd.read_csv("combined_emotion.csv")

# Use medium dataset (better than 10k, faster than full)
data = data.sample(n=15000, random_state=42).reset_index(drop=True)

# =============================
# CLEAN TEXT
# =============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["sentence"] = data["sentence"].apply(clean_text)

texts = data["sentence"]
labels = data["emotion"]

# =============================
# LABEL ENCODING
# =============================
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# =============================
# TOKENIZATION
# =============================
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

# =============================
# PADDING
# =============================
max_len = 60
X = pad_sequences(sequences, maxlen=max_len)
y = labels

# =============================
# TRAIN-TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# TENSORS
# =============================
X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# =============================
# DATALOADER
# =============================
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

num_classes = len(set(y))

# =============================
# MODEL (UPGRADED)
# =============================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

model = LSTMModel()

# =============================
# TRAINING SETUP
# =============================
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5
)

epochs = 15

# =============================
# TRAINING LOOP
# =============================
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =============================
# EVALUATION (ONLY ACCURACY)
# =============================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

acc = accuracy_score(all_labels, all_preds)

print("\n Final Accuracy:", round(acc * 100, 2), "%")

# =============================
# SAVE MODEL
# =============================
torch.save(model.state_dict(), "model.pth")
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

print("\n✅ Training Completed & Model Saved")