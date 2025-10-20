# %%
import torch
import torch.nn as nn
import torch.optim
import pandas as pd

# %%

model = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Dropout(0.2),  
    nn.Linear(128, 64),
    nn.ReLU(), 
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# %%
df=pd.read_csv('sleep_score_data_fitbit.csv')
features=df[['deep_sleep_in_minutes','resting_heart_rate','restlessness']]
target=df[['overall_score']]

# %%
X=torch.FloatTensor(features.values)
y=torch.FloatTensor(target.values)

# %%
total_samples=len(df)
train_size=int(0.8*total_samples)
indices=torch.randperm(total_samples)

# %%
x_train=X[indices][:train_size]
x_test=X[indices][train_size:]
y_train=y[indices][:train_size]
y_test=y[indices][train_size:]

# %%
loss_function=nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# %%
for epoch in range(2500):
    forwardpass=model(x_train).squeeze()
    loss=loss_function(forwardpass,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if epoch % 100 == 0:
    print(f'Epoch {epoch}, Loss: {loss.item()}')


# %%

print("\nResults:")
with torch.no_grad():
    predictions = model(X).squeeze()
    for i in range(len(df)):
        print(f"Actual: {y[i].item()}, Predicted: {predictions[i].item():.2f}")


