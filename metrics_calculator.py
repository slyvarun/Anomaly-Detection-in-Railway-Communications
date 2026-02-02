import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

class MetricsCalculator:
    def __init__(self, labels):
        self.labels = labels
        self.precision = []
        self.recall = []
        self.fscore = []
        self.accuracy = []

        self.metrics_df = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
        self.class_report_df = pd.DataFrame()
        self.class_performance_dfs = {}

        if not os.path.exists('results'):
            os.makedirs('results')

    def calculate_metrics(self, algorithm, predict, y_test, y_score=None):
        categories = self.labels

        # Calculate overall metrics
        a = accuracy_score(y_test, predict) * 100
        p = precision_score(y_test, predict, average='macro') * 100
        r = recall_score(y_test, predict, average='macro') * 100
        f = f1_score(y_test, predict, average='macro') * 100

        # Append to lists
        self.accuracy.append(a)
        self.precision.append(p)
        self.recall.append(r)
        self.fscore.append(f)

        # Append to DataFrame
        metrics_entry = pd.DataFrame({
            'Algorithm': [algorithm],
            'Accuracy': [a],
            'Precision': [p],
            'Recall': [r],
            'F1-Score': [f]
        })
        self.metrics_df = pd.concat([self.metrics_df, metrics_entry], ignore_index=True)

        print(f"{algorithm} Accuracy  : {a:.2f}")
        print(f"{algorithm} Precision : {p:.2f}")
        print(f"{algorithm} Recall    : {r:.2f}")
        print(f"{algorithm} FScore    : {f:.2f}")

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, predict)

        # Classification report
        CR = classification_report(y_test, predict, target_names=[str(c) for c in categories], output_dict=True)
        print(f"{algorithm} Classification Report")
        print(f"{algorithm}\n{classification_report(y_test, predict, target_names=[str(c) for c in categories])}\n")

        # Classification report dataframe
        cr_df = pd.DataFrame(CR).transpose()
        cr_df['Algorithm'] = algorithm
        self.class_report_df = pd.concat([self.class_report_df, cr_df], ignore_index=False)

        # Class-specific performance
        for category in categories:
            class_entry = pd.DataFrame({
                'Algorithm': [algorithm],
                'Precision': [CR[str(category)]['precision'] * 100],
                'Recall': [CR[str(category)]['recall'] * 100],
                'F1-Score': [CR[str(category)]['f1-score'] * 100],
                'Support': [CR[str(category)]['support']]
            })

            if str(category) not in self.class_performance_dfs:
                self.class_performance_dfs[str(category)] = pd.DataFrame(columns=['Algorithm', 'Precision', 'Recall', 'F1-Score', 'Support'])

            self.class_performance_dfs[str(category)] = pd.concat([self.class_performance_dfs[str(category)], class_entry], ignore_index=True)

        # Plot confusion matrix
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(conf_matrix, xticklabels=categories, yticklabels=categories, annot=True, cmap="viridis", fmt="g")
        ax.set_ylim([0, len(categories)])
        plt.title(f"{algorithm} Confusion Matrix")
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.savefig(f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png")
        plt.show()

        # ROC Curve Plot
        if y_score is None:
                print("[WARNING] y_score is None. Cannot plot ROC.")
                return

        os.makedirs("results", exist_ok=True)

        n_classes = len(categories)

        plt.figure(figsize=(10, 8))

        # Binary classification
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])  # probability for class 1
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {categories[1]} (AUC = {roc_auc:.2f})")

        # Multiclass classification
        else:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            fpr, tpr, roc_auc = dict(), dict(), dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], label=f'Class {categories[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.title(f"{algorithm} ROC Curve{'s' if n_classes > 2 else ''} (One-vs-Rest)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
        plt.show()


    def plot_classification_graphs(self):
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

        os.makedirs('results', exist_ok=True)
        plt.tight_layout()
        plt.savefig('results/classifier_performance.png')
        plt.show()

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

        melted_df_new = self.metrics_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
        melted_df_new = melted_df_new.round(3)

        return melted_df_new
