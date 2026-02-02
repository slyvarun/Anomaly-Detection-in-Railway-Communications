import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class GraphPlotter:
    def __init__(self, metrics_df, class_performance_dfs):
        self.metrics_df = metrics_df
        self.class_performance_dfs = class_performance_dfs
        
        os.makedirs('results', exist_ok=True)

    def plot_overall_metrics(self):
        melted_df = pd.melt(self.metrics_df, id_vars=['Algorithm'],
                            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            var_name='Parameters', value_name='Value')
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Parameters', y='Value', hue='Algorithm', data=melted_df)
        
        plt.title('Classifier Performance Comparison', fontsize=14, pad=10)
        plt.ylabel('Score (%)', fontsize=12)
        plt.xlabel('Metrics', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)
        
        plt.tight_layout()
        plt.savefig('results/classifier_performance.png')
        plt.show()

    def plot_class_specific_metrics(self):
        for class_name, class_df in self.class_performance_dfs.items():
            melted_class_df = pd.melt(class_df, id_vars=['Algorithm'],
                                      value_vars=['Precision', 'Recall', 'F1-Score'],
                                      var_name='Parameters', value_name='Value')

            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Parameters', y='Value', hue='Algorithm', data=melted_class_df)
            
            plt.title(f'Class {class_name} Performance Comparison', fontsize=14, pad=10)
            plt.ylabel('Score (%)', fontsize=12)
            plt.xlabel('Metrics', fontsize=12)
            plt.xticks(rotation=0)
            plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', padding=3)
            
            plt.tight_layout()
            plt.savefig(f'results/class_{class_name}_performance.png')
            plt.show()

    def plot_all(self):
        self.plot_overall_metrics()
        self.plot_class_specific_metrics()
