"""
loop.py

Defines the core training loop for a Vision Transformer (ViT)-based image classification model.
Handles training, evaluation, learning rate scheduling, metric logging, and early stopping.

Key Features:
- Uses CrossEntropyLoss and Adam optimizer
- Supports ReduceLROnPlateau scheduler and early stopping
- Logs metrics including loss, accuracy, F1, precision, recall, and gradient norms
- Integrates with MLflow via a custom ExperimentManager
"""

from tqdm import tqdm
import torch
from colorama import Fore, Style
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score


def train_loop(
        model,
        train_loader,
        test_loader,
        experiment_manager,
        epochs=5,
        lr=0.001,
        patience=2,
        min_delta=0.0001,
        batch_size=32,
        device=None
):
    """
    Train a PyTorch model using the specified parameters and log metrics throughout.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): Dataloader for training data.
        test_loader (DataLoader): Dataloader for validation/testing data.
        experiment_manager (ExperimentManager): Wrapper to manage MLflow logging and artifacts.
        epochs (int): Number of epochs to train.
        lr (float): Initial learning rate.
        patience (int): Number of epochs with no improvement to wait before early stopping.
        min_delta (float): Minimum change in validation loss to qualify as improvement.
        batch_size (int): Batch size used for training.
        device (torch.device or None): Device to use (auto-detected if None).

    Returns:
        dict: Training results including losses, accuracies, metrics, and best model state.
    """

    # --- PyTorch Runtime Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)  # Reproducibility
    model.to(device)
    print(f"Model is on device: {Fore.CYAN}{next(model.parameters()).device}{Style.RESET_ALL}")

    experiment_manager.logger.info("Training started.")
    experiment_manager.log_params({
        "optimizer": "Adam",
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": len(train_loader),
        "patience": patience,
        "min_delta": min_delta,
        "loss_fn": "CrossEntropyLoss",
    })

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    train_losses, eval_losses, eval_accuracies = [], [], []
    epoch_weights, epoch_biases = [], []

    all_preds, all_labels = [], []

    best_loss = float('inf')
    best_epoch = 0
    trigger_times = 0

    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for X_batch, y_batch in pbar:
            try:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()

                # Log gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                experiment_manager.log_metrics({"gradient_norm": total_norm}, step=epoch)

                optimizer.step()
                epoch_train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            except Exception as e:
                experiment_manager.logger.error(f"Training batch exception: {e}")
                continue

        train_losses.append(epoch_train_loss / len(train_loader))

        # --- Evaluation ---
        try:
            model.eval()
            correct = 0
            total = 0
            epoch_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    epoch_val_loss += loss_fn(y_pred, y_batch).item()
                    correct += (y_pred.argmax(1) == y_batch).sum().item()
                    total += y_batch.size(0)
                    all_preds.extend(y_pred.argmax(1).cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

            mean_val_loss = epoch_val_loss / len(test_loader)
            epoch_accuracy = correct / total * 100

            experiment_manager.log_metrics({
                "val_loss": mean_val_loss,
                "val_accuracy": epoch_accuracy
            }, step=epoch)

            eval_losses.append(mean_val_loss)
            eval_accuracies.append(epoch_accuracy)

            f1 = f1_score(all_labels, all_preds, average='weighted')
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')

            experiment_manager.log_metrics({
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            }, step=epoch)

            if epoch_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_accuracy
                best_model_state = model.state_dict()

            # Early stopping logic
            if mean_val_loss < best_loss - min_delta:
                best_loss = mean_val_loss
                best_epoch = epoch
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    experiment_manager.logger.warning("Early stopping triggered.")
                    break

            # Learning rate tracking and scheduling
            current_lr = scheduler.get_last_lr()[0]
            experiment_manager.log_metrics({"learning_rate": current_lr}, step=epoch)
            scheduler.step(mean_val_loss)

        except Exception as e:
            experiment_manager.logger.error(f"Exception during validation: {e}")
            continue

    # --- Final Accuracy Calculation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            correct += (y_pred.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)

    final_test_accuracy = correct / total * 100
    final_f1 = f1_score(all_labels, all_preds, average='weighted')
    final_precision = precision_score(all_labels, all_preds, average='weighted')
    final_recall = recall_score(all_labels, all_preds, average='weighted')

    return {
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'eval_accuracies': eval_accuracies,
        'best_val_accuracy': best_val_accuracy,
        'final_test_accuracy': final_test_accuracy,
        'final_f1': final_f1,
        'final_precision': final_precision,
        'final_recall': final_recall,
        'epoch_weights': epoch_weights,
        'epoch_biases': epoch_biases,
        'best_loss': best_loss,
        'best_model_state': best_model_state,
        'best_epoch': best_epoch,
        'trigger_times': trigger_times,
        'all_preds': all_preds,
        'all_labels': all_labels,
    }
