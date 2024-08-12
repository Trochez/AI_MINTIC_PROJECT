import ultralytics
from ultralytics import YOLO
import shutil, os
import torch


savedModel = YOLO('/home/trocha/ai_mintic/best.torchscript')
torch.save(savedModel, "./game_hands.pt")
model = torch.load("./game_hands.pt")


#item = savedModel.predict('/home/trocha/ai_mintic/2024-08-10-204059.jpg', verbose=False, save=False, conf=0.5)
item = model.predict('/home/trocha/ai_mintic/2024-08-10-204059.jpg', verbose=False, save=False, conf=0.5)


print("item ---")
print(item)
