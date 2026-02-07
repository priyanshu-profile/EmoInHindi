import torch
import numpy as np
from training.metrics import compute_multilabel_metrics


@torch.no_grad()
def evaluate(model, dataloader, device, threshold=0.5):
    """
    Evaluation for multi-label emotion + intensity
    """
    model.eval()

    all_emotion_preds = []
    all_emotion_labels = []
    all_intensity_preds = []
    all_intensity_labels = []

    for batch in dataloader:
        inputs = batch["context_embeddings"].to(device)
        emotion_labels = batch["emotion_labels"].cpu().numpy()
        intensity_labels = batch["intensity_labels"].cpu().numpy()

        emotion_preds, intensity_preds = model(inputs)

        all_emotion_preds.append(emotion_preds.cpu().numpy())
        all_emotion_labels.append(emotion_labels)

        all_intensity_preds.append(intensity_preds.argmax(dim=-1).cpu().numpy())
        all_intensity_labels.append(intensity_labels)

    all_emotion_preds = np.vstack(all_emotion_preds)
    all_emotion_labels = np.vstack(all_emotion_labels)

    metrics = compute_multilabel_metrics(
        all_emotion_labels,
        all_emotion_preds,
        threshold=threshold
    )

    intensity_accuracy = np.mean(
        np.concatenate(all_intensity_preds) ==
        np.concatenate(all_intensity_labels)
    )

    metrics["intensity_accuracy"] = intensity_accuracy
    return metrics
