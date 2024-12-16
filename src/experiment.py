# src/experiments.py

import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.flower_preprocessor import get_data_loaders
from models.flower_classification import FlowersClassifier, train, evaluate

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def run_experiment(model_class, num_units, learning_rate, optimizer_class, num_epochs=2):
    # Set up data loaders
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'data', 'external', 'flowers', 'train')
    test_dir = os.path.join(base_dir, 'data', 'external', 'flowers', 'test')

    train_loader, val_loader, _, num_classes = get_data_loaders(train_dir, test_dir, batch_size=64, val_split=0.2)

    # Set up model and training
    model = model_class(num_classes=num_classes, num_units=num_units).to(device)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(f'runs/experiment_{num_units}_{learning_rate}_{optimizer_class.__name__}')

    best_val_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, writer, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, writer, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    writer.close()
    return best_val_acc


def visualize_results(results_exp1, results_exp2):
    # Experiment 1 visualization
    df1 = pd.DataFrame(results_exp1, columns=['Units', 'Learning Rate', 'Accuracy'])
    plt.figure(figsize=(8, 6))
    pivot_table = df1.pivot('Units', 'Learning Rate', 'Accuracy')
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Experiment 1: Units vs Learning Rate')
    plt.savefig('experiment1_results.png')
    plt.close()

    # Experiment 2 visualization
    df2 = pd.DataFrame(results_exp2, columns=['Optimizer', 'Learning Rate', 'Accuracy'])
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Optimizer', y='Accuracy', hue='Learning Rate', data=df2)
    plt.title('Experiment 2: Optimizers and Learning Rates')
    plt.savefig('experiment2_results.png')
    plt.close()


def create_summary_report():
    report = """
    Flower Classification Model Experiments

    Experiment 1: Varying Units and Learning Rate
    [Insert experiment1_results.png here]

    Observation: [Brief description of the results]

    Experiment 2: Comparing Optimizers and Learning Rates
    [Insert experiment2_results.png here]

    Observation: [Brief description of the results]

    Reflection:
    [1-2 sentences explaining the results from a theoretical perspective]
    [1-2 sentences on how the findings align with or differ from your initial hypotheses]

    Model Architecture:
    [Insert a text description or visual representation of your FlowersClassifier architecture]

    Conclusion:
    [1-2 sentences summarizing the key insights from the experiments]
    """

    with open('experiment_summary.txt', 'w') as f:
        f.write(report)


def main():
    print(f"Using device: {device}")

    # Experiment 1: Vary number of units and learning rate
    units_list = [32, 64, 128]
    lr_list = [0.001, 0.0001, 0.00001]
    results_exp1 = []

    for units in units_list:
        for lr in lr_list:
            print(f"Running experiment with units={units}, lr={lr}")
            acc = run_experiment(FlowersClassifier, units, lr, optim.Adam)
            results_exp1.append((units, lr, acc))

    # Experiment 2: Compare optimizers and learning rates
    optimizer_list = [optim.Adam, optim.SGD, optim.RMSprop]
    results_exp2 = []

    for opt in optimizer_list:
        for lr in lr_list:
            print(f"Running experiment with optimizer={opt.__name__}, lr={lr}")
            acc = run_experiment(FlowersClassifier, 64, lr, opt)
            results_exp2.append((opt.__name__, lr, acc))

    # Visualize results
    visualize_results(results_exp1, results_exp2)

    # Create summary report
    create_summary_report()

    print("Experiments completed. Please check 'experiment_summary.txt' for results.")


if __name__ == "__main__":
    main()
