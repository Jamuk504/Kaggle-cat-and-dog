import torch
import os
import sys
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import model
from config import valid_transform, MODEL_TO_TEST, DEVICE, NUM_WORKERS, MISCLASSIFIED_COUNT, CLASSES, IMAGES_PER_ROW


def evaluate_model(model, data_loader):
    correct = 0
    total = 0

    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    model.eval()

    with torch.no_grad():
        print(f"Starting evaluation on the test dataset:")

        for i, data in enumerate(data_loader):
            if i % 10==0:
                print(f"Processed batch {i}:")

            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if len(misclassified_images) < MISCLASSIFIED_COUNT:
                incorrect_indices = torch.where(predicted != labels)[0]

                for idx in incorrect_indices:
                    if len(misclassified_images) >= MISCLASSIFIED_COUNT:
                        break

                    misclassified_images.append(inputs[idx].cpu())
                    misclassified_labels.append(labels[idx].item())
                    misclassified_preds.append(predicted[idx].item())

    accuracy = 100 * correct / total
    return accuracy, total, misclassified_images, misclassified_labels, misclassified_preds

def imshow(img, misclassified_labels, misclassified_preds):
    num_rows = (len(misclassified_labels) + IMAGES_PER_ROW - 1) // IMAGES_PER_ROW

    fig_width = IMAGES_PER_ROW * 2
    fig_height = num_rows * 2
    fig = plt.figure(figsize=(fig_width, fig_height))

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    npimg = img.numpy()
    npimg = (npimg * std) + mean
    npimg = np.clip(npimg, 0, 1)
    npimg = (npimg * 255).astype(np.uint8)

    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=None)
    plt.title("Misclassified Samples")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_set = datasets.ImageFolder(root='data/valid', transform=valid_transform)

    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    model.to(DEVICE)

    if not os.path.exists(MODEL_TO_TEST):
        print(f"ERROR: Model file not found at {MODEL_TO_TEST}")
        print("Please set the correct path in the MODEL_TO_TEST variable and ensure the file exists")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(MODEL_TO_TEST, map_location=DEVICE))
    except Exception as e:
        print(f"ERROR: Could not load model state dict from {MODEL_TO_TEST}. Error: {e}")
        sys.exit(1)

    print(f"Model successfully loaded from {MODEL_TO_TEST} and set to evaluation mode")

    accuracy, total_tested, misclassified_images, misclassified_labels, misclassified_preds = \
        evaluate_model(model, test_loader)
    print("\nMODEL EVALUATION SUMMARY:")
    print(f"Total Examples Tested: {total_tested}")
    print(f"Overall Accuracy: {accuracy:.2f} %")

    if misclassified_images:
        print("\nMISCLASSIFIED SAMPLES")
        
        images_to_display = misclassified_images[:MISCLASSIFIED_COUNT]
        labels_to_display = misclassified_labels[:MISCLASSIFIED_COUNT]
        preds_to_display = misclassified_preds[:MISCLASSIFIED_COUNT]
        
        img_to_show = torch.stack(images_to_display)
        
        grid_image = torchvision.utils.make_grid(img_to_show, nrow=IMAGES_PER_ROW)
        
        imshow(grid_image, labels_to_display, preds_to_display)
        
        print(f"Showing the first {len(images_to_display)} misclassified predictions:")
        print("  {:12s} | {:12s} ".format("Ground Truth", "Predicted"))
        
        for gt, pred in zip(labels_to_display, preds_to_display):
            gt_label = CLASSES[gt]
            pred_label = CLASSES[pred]
            marker = "X"
            print("  {:12s} | {:12s} {}".format(gt_label, pred_label, marker))
    else:
        print("No misclassified images were collected.")