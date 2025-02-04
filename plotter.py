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
    Plot training and evaluation metrics from output directory.
    Creates PDF with separate plots for each metric over training steps.
    Uses a modern, professional style with custom color palette.
    """
    # Load metrics history
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics_history = json.load(f)

    # Set style and color palette
    plt.style.use('bmh')  # Using 'bmh' style which is a modern, clean style
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    # Create PDF to save all plots
    pdf_path = os.path.join(output_dir, 'training_plots.pdf')
    with PdfPages(pdf_path) as pdf:
        
        # Plot loss
        plt.figure(figsize=(12,7))
        steps = [int(x) for x in metrics_history['loss'].keys()]
        values = list(metrics_history['loss'].values())
        
        # Plot raw data with low alpha
        plt.plot(steps, values, color=colors[0], alpha=0.3, linewidth=1.5, label='Raw data')
        
        # Calculate and plot moving average
        ma_values = moving_average(values)
        ma_steps = steps[len(steps)-len(ma_values):]
        plt.plot(ma_steps, ma_values, color=colors[0], linewidth=2.5, label='Moving average')
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Plot training metrics
        train_metrics = metrics_history['train_metrics']
        if train_metrics:
            # Get all metric names from first entry
            first_step = list(train_metrics.keys())[0]
            metric_names = ['accuracy', 'total_score', 'correctness', 'int_format', 
                          'strict_format', 'soft_format', 'xml_count']
            
            for metric, color in zip(metric_names, colors):
                plt.figure(figsize=(12,7))
                steps = [int(x) for x in train_metrics.keys()]
                values = [metrics[metric] for metrics in train_metrics.values()]
                
                # Plot raw data with low alpha
                plt.plot(steps, values, color=color, alpha=0.3, linewidth=1.5, label='Raw data')
                
                # Calculate and plot moving average
                ma_values = moving_average(values)
                ma_steps = steps[len(steps)-len(ma_values):]
                plt.plot(ma_steps, ma_values, color=color, linewidth=2.5, label='Moving average')
                
                plt.xlabel('Training Steps', fontsize=12)
                plt.ylabel(f'{metric.replace("_", " ").title()} Score', fontsize=12)
                plt.title(f'Training {metric.replace("_", " ").title()}', fontsize=14, pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    plot_metrics("9_run")
