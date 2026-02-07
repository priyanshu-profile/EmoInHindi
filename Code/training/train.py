import torch
from torch.optim import Adam
from tqdm import tqdm

from training.evaluate import evaluate


def train(
    model,
    train_loader,
    val_loader,
    multitask_loss,
    emotion_criterion,
    intensity_criterion,
    device,
    epochs=30,
    lr=0.003
):
    """
    Training loop for EmoInHindi model
    """

    optimizer = Adam(
        list(model.parameters()) + list(multitask_loss.parameters()),
        lr=lr
    )

    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in progress:
            optimizer.zero_grad()

            inputs = batch["context_embeddings"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            intensity_labels = batch["intensity_labels"].to(device)

            emotion_preds, intensity_preds = model(inputs)

            loss_emotion = emotion_criterion(emotion_preds, emotion_labels)
            loss_intensity = intensity_criterion(intensity_preds, intensity_labels)

            loss = multitask_loss(loss_emotion, loss_intensity)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)

        val_metrics = evaluate(model, val_loader, device)
        micro_f1 = val_metrics["micro_f1"]

        print(
            f"\nEpoch {epoch} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Micro-F1: {micro_f1:.4f} | "
            f"HL: {val_metrics['hamming_loss']:.4f} | "
            f"JI: {val_metrics['jaccard_index']:.4f} | "
            f"Intensity Acc: {val_metrics['intensity_accuracy']:.4f}"
        )

        if micro_f1 > best_f1:
            best_f1 = micro_f1
            torch.save(model.state_dict(), "best_model.pt")
            print("✔ Best model saved")

    print(f"\nTraining completed. Best Micro-F1: {best_f1:.4f}")
