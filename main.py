import os.path

from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm
from transformers import AdamW
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import pandas as pd
from torch.utils.data import DataLoader
import fire
from dataset import ImageSegmentationDataset


def train(epochs: int = 10, data_path: str = 'dataset', batch_size: int = 2):
    root_dir = data_path
    feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False, size=128)

    train_dataset = ImageSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, transforms=None)
    valid_dataset = ImageSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, transforms=None,
                                             train=False)
    classes = pd.read_csv(os.path.join(data_path, 'labels.csv'))['label_list']
    id2label = classes.to_dict()
    id2label[0] = 'back_ground'
    label2id = {v: k for k, v in id2label.items()}
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", ignore_mismatched_sizes=True,
                                                             num_labels=len(id2label), id2label=id2label,
                                                             label2id=label2id,
                                                             reshape_last_stage=True)
    optimizer = AdamW(model.parameters(), lr=0.00006)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        pbar = tqdm(train_dataloader)
        accuracies = []
        losses = []
        val_accuracies = []
        val_losses = []
        model.train()
        for idx, batch in enumerate(pbar):
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(pixel_values=pixel_values, labels=labels)

            # evaluate
            upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear",
                                                         align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            mask = (labels != 0)  # we don't include the background class in the accuracy calculation
            pred_labels = predicted[mask].detach().cpu().numpy()
            true_labels = labels[mask].detach().cpu().numpy()
            accuracy = accuracy_score(pred_labels, true_labels)
            loss = outputs.loss
            accuracies.append(accuracy)
            losses.append(loss.item())
            pbar.set_postfix(
                {'Batch': idx, 'Pixel-wise accuracy': sum(accuracies) / len(accuracies),
                 'Loss': sum(losses) / len(losses)})

            # backward + optimize
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(valid_dataloader):
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:],
                                                                 mode="bilinear",
                                                                 align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)

                    mask = (labels != 0)  # we don't include the background class in the accuracy calculation
                    pred_labels = predicted[mask].detach().cpu().numpy()
                    true_labels = labels[mask].detach().cpu().numpy()
                    accuracy = accuracy_score(pred_labels, true_labels)
                    val_loss = outputs.loss
                    val_accuracies.append(accuracy)
                    val_losses.append(val_loss.item())

        print(f"Train Pixel-wise accuracy: {sum(accuracies) / len(accuracies)}\
             Train Loss: {sum(losses) / len(losses)}\
             Val Pixel-wise accuracy: {sum(val_accuracies) / len(val_accuracies)}\
             Val Loss: {sum(val_losses) / len(val_losses)}")


if __name__ == '__main__':
    fire.Fire(train)
