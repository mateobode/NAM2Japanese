import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def load_csv_data(file_path):
    """Load CSV data and return steps and values."""
    df = pd.read_csv(file_path)
    return df['Step'].values, df['Value'].values

def create_training_plots():
    """Create comprehensive training plots from train_data/ directory."""
    train_data_dir = "train_data"
    plots_dir = "plots"
    
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    print("Creating loss comparison plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    train_steps, train_loss = load_csv_data(os.path.join(train_data_dir, "train_loss.csv"))
    eval_steps, eval_loss = load_csv_data(os.path.join(train_data_dir, "eval_loss.csv"))
    
    ax.plot(train_steps, train_loss, color='#2E86C1', linewidth=2, label='Training Loss', alpha=0.8)
    ax.plot(eval_steps, eval_loss, color='#E74C3C', linewidth=3, label='Evaluation Loss', marker='o', markersize=4)
    
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training vs Evaluation Loss', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    final_train_loss = train_loss[-1]
    final_eval_loss = eval_loss[-1]
    ax.text(0.02, 0.98, f'Final Training Loss: {final_train_loss:.4f}\\nFinal Evaluation Loss: {final_eval_loss:.4f}', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Creating WER plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    wer_steps, wer_values = load_csv_data(os.path.join(train_data_dir, "eval_wer.csv"))
    
    ax.plot(wer_steps, wer_values, color='#8E44AD', linewidth=2.5, marker='s', markersize=5, label='Evaluation WER')
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Word Error Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Word Error Rate During Training', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    final_wer = wer_values[-1]
    ax.text(0.02, 0.98, f'Final WER: {final_wer:.2f}%', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'wer_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Creating gradient norm plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    grad_steps, grad_norms = load_csv_data(os.path.join(train_data_dir, "grad_norm.csv"))
    
    ax.plot(grad_steps, grad_norms, color='#F39C12', linewidth=1.5, alpha=0.7, label='Gradient Norm')

    window_size = 100
    if len(grad_norms) > window_size:
        moving_avg = np.convolve(grad_norms, np.ones(window_size)/window_size, mode='valid')
        moving_avg_steps = grad_steps[window_size-1:]
        ax.plot(moving_avg_steps, moving_avg, color='#D35400', linewidth=3, label=f'Moving Average ({window_size} steps)')
    
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontsize=14, fontweight='bold')
    ax.set_title('Gradient Norm During Training', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'gradient_norm.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Creating learning rate plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lr_steps, lr_values = load_csv_data(os.path.join(train_data_dir, "learning_rate.csv"))
    
    ax.plot(lr_steps, lr_values, color='#27AE60', linewidth=2.5, label='Learning Rate')
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold', pad=20)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.grid(True, alpha=0.3)
    
    min_lr = np.min(lr_values)
    max_lr = np.max(lr_values)
    ax.text(0.02, 0.98, f'Max LR: {max_lr:.2e}\\nMin LR: {min_lr:.2e}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Creating combined overview plot...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss subplot
    ax1.plot(train_steps, train_loss, color='#2E86C1', linewidth=2, label='Training Loss')
    ax1.plot(eval_steps, eval_loss, color='#E74C3C', linewidth=2, label='Evaluation Loss', marker='o', markersize=3)
    ax1.set_title('Loss Comparison', fontweight='bold')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # WER subplot
    ax2.plot(wer_steps, wer_values, color='#8E44AD', linewidth=2, marker='s', markersize=3)
    ax2.set_title('Word Error Rate', fontweight='bold')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('WER (%)')
    ax2.grid(True, alpha=0.3)
    
    # Gradient Norm subplot
    ax3.plot(grad_steps, grad_norms, color='#F39C12', linewidth=1, alpha=0.6)
    if len(grad_norms) > window_size:
        ax3.plot(moving_avg_steps, moving_avg, color='#D35400', linewidth=2)
    ax3.set_title('Gradient Norm', fontweight='bold')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Gradient Norm')
    ax3.grid(True, alpha=0.3)
    
    # Learning Rate subplot
    ax4.plot(lr_steps, lr_values, color='#27AE60', linewidth=2)
    ax4.set_title('Learning Rate', fontweight='bold')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Learning Rate')
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics Overview', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\\nAll plots saved to '{plots_dir}/' directory:")
    print("- loss_comparison.png: Training vs Evaluation Loss")
    print("- wer_progress.png: Word Error Rate progression")
    print("- gradient_norm.png: Gradient norm with moving average")
    print("- learning_rate.png: Learning rate schedule")
    print("- training_overview.png: Combined overview of all metrics")

if __name__ == "__main__":
    create_training_plots()