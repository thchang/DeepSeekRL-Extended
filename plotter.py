import json
import os
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def moving_average(data, window_size=5):
    """Calculate moving average with given window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def plot_metrics(output_dir):
    """
    Plot training metrics from training_logs directory.
    Creates PDF with separate plots for each metric over training steps.
    Uses a modern, professional style with custom color palette.
    """
    # Load training logs
    train_logs_path = os.path.join(output_dir, 'training_logs', 'train_logs.json')
    with open(train_logs_path, 'r') as f:
        train_logs = json.load(f)

    # Set style and color palette
    plt.style.use('bmh')  # Using 'bmh' style which is a modern, clean style
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    # Create PDF to save all plots
    pdf_path = os.path.join(output_dir, 'training_plots.pdf')
    with PdfPages(pdf_path) as pdf:
        
        # Get all metric names from first entry
        first_step = list(train_logs.keys())[0]
        metric_names = list(train_logs[first_step].keys())
        
        # Plot main metrics
        for metric, color in zip(metric_names, colors):
            plt.figure(figsize=(12,7))
            steps = [int(x) for x in train_logs.keys()]
            values = [metrics[metric] for metrics in train_logs.values()]
            
            # Plot raw data with low alpha
            plt.plot(steps, values, color=color, alpha=0.3, linewidth=1.5, label='Raw data')
            
            # Calculate and plot moving average if we have enough data points
            if len(values) > 5:
                ma_values = moving_average(values)
                ma_steps = steps[len(steps)-len(ma_values):]
                plt.plot(ma_steps, ma_values, color=color, linewidth=2.5, label='Moving average')
            
            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
            plt.title(f'Training {metric.replace("_", " ").title()}', fontsize=14, pad=20)
            plt.grid(True, alpha=0.3)
            plt.legend()
            pdf.savefig(bbox_inches='tight')
            plt.close()

        # Plot total loss
        plt.figure(figsize=(12,7))
        steps = [int(x) for x in train_logs.keys()]
        total_loss = [metrics['total_loss'] for metrics in train_logs.values()]

        plt.plot(steps, total_loss, color='#e74c3c', alpha=0.3, linewidth=1.5, label='Total Loss (Raw)')
        if len(total_loss) > 5:
            ma_total_loss = moving_average(total_loss)
            ma_steps = steps[len(steps)-len(ma_total_loss):]
            plt.plot(ma_steps, ma_total_loss, color='#e74c3c', linewidth=2.5, label='Total Loss (MA)')

        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title('Total Loss', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Plot KL loss separately
        plt.figure(figsize=(12,7))
        kl_loss = [metrics['mean_kl_loss'] for metrics in train_logs.values()]

        plt.plot(steps, kl_loss, color='#3498db', alpha=0.3, linewidth=1.5, label='KL Loss (Raw)')
        if len(kl_loss) > 5:
            ma_kl_loss = moving_average(kl_loss)
            ma_steps = steps[len(steps)-len(ma_kl_loss):]
            plt.plot(ma_steps, ma_kl_loss, color='#3498db', linewidth=2.5, label='KL Loss (MA)')

        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title('KL Loss', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    plot_metrics("final")
