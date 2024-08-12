import torch

model = torch.load("../game_hands.pt")

def predict_image(filepath):
   
    predictions = model.predict(filepath)
    
    return predictions