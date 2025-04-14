import torch 
import torchvision.transforms as transforms 
import torch.optim as optim
import torchvision.transforms.functional as FT 
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import(
    intersection_over_union,
    non_max_suppression, 
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint, 
    load_checkpoint,
)

from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

#hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
DROPOUT = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "/cs/student/suryagunukula/CMPSC 190I/PascalVOCDataset/images"
LABEL_DIR = "/cs/student/suryagunukula/CMPSC 190I/PascalVOCDataset/labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        
        return img, bboxes
    
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave = True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Update progress bar
        loop.set_postfix(loss = loss.item())

    avg_loss = sum(mean_loss)/len(mean_loss)
    print(f"Mean loss was {avg_loss}")
    return avg_loss
    

def main():
    model = Yolov1(split_size = 7, num_boxes = 2, num_classes = 20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    train_dataset = VOCDataset(
        "/cs/student/suryagunukula/CMPSC 190I/PascalVOCDataset/train.csv", #do train.csv next
        transform = transform,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR,
    )
    
    test_dataset = VOCDataset(
        "/cs/student/suryagunukula/CMPSC 190I/PascalVOCDataset/test.csv",
        transform = transform,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        shuffle = True,
        drop_last = False,
    )

    loss_history = []
    map_history = []
    best_map = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Always train
        mean_loss = train_fn(train_loader, model, optimizer, loss_fn)
        loss_history.append(mean_loss)

        # Only calculate mAP every 5 epochs or final epoch
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.05
            )
            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Train mAP: {mean_avg_prec}")
            map_history.append(mean_avg_prec)

            if mean_avg_prec > best_map:
                print("Saving BEST MAP")
                best_map = mean_avg_prec
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename="checkpoints/best_map.pth.tar")
        else:
            map_history.append(torch.tensor(0.0))  # placeholder if mAP not calculated

        torch.save(torch.tensor(loss_history), "metrics/loss.pt")
        torch.save(torch.tensor(map_history), "metrics/mAP.pt")




if __name__ == "__main__":
    main()
