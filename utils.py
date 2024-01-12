import torch
from dataset import SeeTrough2d
from dataset import SeeTrough3d
from torch.utils.data import DataLoader
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
 
def normalize(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.nan_to_num(img, 0)
    return img

def get_loaders(
    train_x,
    train_y,
    # val_x,
    # val_y,
    batch_size,
    # train_transform,
    # val_transform,
    num_workers=1,
    pin_memory=True,
):
    train_ds = SeeTrough2d(
        img_dir_x=train_x,
        img_dir_y=train_y,
        # transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # val_ds = MayoDataset(
    #     img_dir_x=val_x,
    #     img_dir_y=val_y,
    #     # transform=val_transform,
    # )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=1,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=False,
    # )

    # should rerturn val_loader
    return train_loader

# def check_accuracy(loader, model, device="cuda"):
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y.to(device).unsqueeze(1)
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)
#             dice_score += (2 * (preds * y).sum()) / (
#                 (preds + y).sum() + 1e-8
#             )

#     print(
#         f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
#     )
#     print(f"Dice score: {dice_score/len(loader)}")
#     model.train()
