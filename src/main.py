import os
import time
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data.flower_preprocessor import get_data_loaders
from models.flower_classification import FlowersClassifier, train, evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from torchsummary import summary


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def get_model_info(model, input_size):
    # Move model to CPU for summary
    model_cpu = model.to('cpu')

    # Get model summary
    model_summary = summary(model_cpu, input_size, device='cpu')

    # Move model back to original device
    model.to(device)

    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model_summary, total_params, trainable_params


def create_summary_table(model_info, hyperparameters, performance_metrics):
    data = {
        'Architecture': [model_info['architecture']],
        'Total Parameters': [model_info['total_params']],
        'Trainable Parameters': [model_info['trainable_params']],
        'Input Size': [str(model_info['input_size'])],
        'Batch Size': [hyperparameters['batch_size']],
        'Learning Rate': [hyperparameters['learning_rate']],
        'Epochs': [hyperparameters['num_epochs']],
        'Optimizer': [hyperparameters['optimizer']],
        'Dropout Rate': [hyperparameters['dropout_rate']],
        'Best Train Accuracy': [performance_metrics['best_train_acc']],
        'Best Validation Accuracy': [performance_metrics['best_val_acc']],
        'Final Train Loss': [performance_metrics['final_train_loss']],
        'Final Validation Loss': [performance_metrics['final_val_loss']],
    }

    df = pd.DataFrame(data)
    return df


def save_summary_table(df, filename):
    df.to_csv(filename, index=False)
    print(f"Summary table saved to {filename}")


def main():
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 10
    batch_size = 128
    learning_rate = 0.00005
    validation_split = 0.2

    # removed dropout due to the size of the dataset
    # dropout_rate = 0.05

    # Get data loaders
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'data', 'external', 'flowers', 'train')
    test_dir = os.path.join(base_dir, 'data', 'external', 'flowers', 'test')

    # Modify get_data_loaders to return train and validation loaders
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(train_dir, test_dir, batch_size=batch_size, val_split=validation_split)


    print("Hyperparameters:")
    print(f"Number of classes: {num_classes}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Validation split: {validation_split}")
    # print(f"Dropout rate: {dropout_rate}")
    print()

    # Initialize the model
    model = FlowersClassifier(num_classes).to(device)

    input_size = (3, 224, 224)
    model_summary, total_params, trainable_params = get_model_info(model, input_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Create SummaryWriter
    writer = SummaryWriter(f'runs/{batch_size}_{learning_rate}_{num_epochs}_reduce-on-plateau')

    # Hyperparameters dict
    hyperparameters = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'optimizer': optimizer.__class__.__name__,
        'dropout_rate': dropout_rate
    }

    # Performance metrics dict
    performance_metrics = {
        'best_train_acc': 0,
        'best_val_acc': 0,
        'final_train_loss': 0,
        'final_val_loss': 0
    }

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        epoch_start_time = time.time()

        train_start_time = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, writer, epoch)
        train_time = time.time() - train_start_time

        # Validation step
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, writer, epoch)

        scheduler.step(val_loss)

        last_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start_time

        # Update best accuracies
        performance_metrics['best_train_acc'] = max(performance_metrics['best_train_acc'], train_acc)
        performance_metrics['best_val_acc'] = max(performance_metrics['best_val_acc'], val_acc)

        # Update final losses
        performance_metrics['final_train_loss'] = train_loss
        performance_metrics['final_val_loss'] = val_loss

        print(f" Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f" Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print(f" Training time: {train_time:.2f}s")
        print(f" Total epoch time: {epoch_time:.2f}s")
        print()

        writer.add_scalar('Learning Rate', last_lr, epoch)

    writer.close()

    model_info = {
        'architecture': type(model).__name__,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'input_size': input_size
    }

    summary_df = create_summary_table(model_info, hyperparameters, performance_metrics)

    # Save the summary table
    summary_filename = os.path.join(base_dir, 'models', 'model_summary.csv')
    save_summary_table(summary_df, summary_filename)

    print("Training finished!")

    # Create directories if they don't exist
    full_models_dir = os.path.join(base_dir, 'models/full_models')
    model_info_dir = os.path.join(base_dir, 'models/model_info')
    os.makedirs(full_models_dir, exist_ok=True)
    os.makedirs(model_info_dir, exist_ok=True)

    # Extract model name
    model_name = writer.log_dir.split('/')[-1]  # Extract the name after 'runs/'

    # Save the full model
    full_model_path = os.path.join(full_models_dir, f'{model_name}_full.pth')
    torch.save(model, full_model_path)
    print(f"Full model saved to {full_model_path}")

    # Save model info and state_dict
    input_channels = 3  # RGB images
    input_height = 224
    input_width = 224
    num_classes = 5  # Number of flower classes

    model_info_path = os.path.join(model_info_dir, f'{model_name}_info.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': type(model).__name__,
        'input_size': (input_channels, input_height, input_width),
        'output_size': num_classes,
    }, model_info_path)
    print(f"Model info and state_dict saved to {model_info_path}")

if __name__ == "__main__":
    main()
