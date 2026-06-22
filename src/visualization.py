"""
Visualization Module
====================
Creates all plots and visualizations for EDA and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Tuple

from .config import config

logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """
    Visualization class for creating all project plots.
    """
    
    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizations will be saved to {self.output_dir}")
    
    def save_figure(self, fig, filename: str, dpi: int = 150):
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {filename}")

    def _has_survived(self, df: pd.DataFrame) -> bool:
        """Check if Survived column exists and is usable"""
        return 'Survived' in df.columns and df['Survived'].notna().any()

    def plot_survival_distribution(self, df: pd.DataFrame):
        """Plot overall survival distribution"""
        if not self._has_survived(df):
            logger.warning("Skipping survival distribution plot: no Survived column")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        
        survival_counts = df['Survived'].value_counts().sort_index()
        colors = ['#ff6b6b', '#51cf66']
        labels = ['Not Survived', 'Survived']
        
        bars = ax.bar(labels, survival_counts.values, color=colors, edgecolor='black')
        ax.set_title('Overall Survival Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Passengers', fontsize=12)
        
        for bar, count in zip(bars, survival_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   str(count), ha='center', fontsize=12, fontweight='bold')
        
        survival_pct = df['Survived'].mean() * 100
        ax.text(0.5, 0.95, f'Survival Rate: {survival_pct:.1f}%',
               transform=ax.transAxes, ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.save_figure(fig, 'survival_distribution.png')
        plt.close()
    
    def plot_age_distribution(self, df: pd.DataFrame):
        """Plot age distribution by survival status"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Overall age distribution
        axes[0].hist(df['Age'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_title('Age Distribution (All Passengers)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(df['Age'].mean(), color='red', linestyle='--', label=f"Mean: {df['Age'].mean():.1f}")
        axes[0].axvline(df['Age'].median(), color='green', linestyle='--', label=f"Median: {df['Age'].median():.1f}")
        axes[0].legend()
        
        # Age by survival (only if Survived exists)
        if self._has_survived(df):
            axes[1].hist([df[df['Survived'] == 0]['Age'].dropna(),
                          df[df['Survived'] == 1]['Age'].dropna()],
                         bins=25, label=['Not Survived', 'Survived'],
                         alpha=0.7, color=['#ff6b6b', '#51cf66'])
            axes[1].set_title('Age Distribution by Survival', fontsize=12, fontweight='bold')
            axes[1].legend()
        else:
            axes[1].hist(df['Age'].dropna(), bins=25, alpha=0.7, color='steelblue')
            axes[1].set_title('Age Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Age')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        self.save_figure(fig, 'age_distribution.png')
        plt.close()
    
    def plot_survival_by_category(self, df: pd.DataFrame):
        """Plot survival rates by different categories"""
        if not self._has_survived(df):
            logger.warning("Skipping survival by category plot: no Survived column")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # By Gender
        survival_by_gender = df.groupby('Sex')['Survived'].mean() * 100
        axes[0, 0].bar(survival_by_gender.index, survival_by_gender.values,
                       color=['#51cf66', '#ff6b6b'])
        axes[0, 0].set_title('Survival Rate by Gender', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Survival Rate (%)')
        axes[0, 0].set_ylim(0, 100)
        for i, v in enumerate(survival_by_gender.values):
            axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # By Passenger Class
        survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
        axes[0, 1].bar(['1st Class', '2nd Class', '3rd Class'], survival_by_class.values,
                       color=['#ffd43b', '#ff922b', '#ff6b6b'])
        axes[0, 1].set_title('Survival Rate by Passenger Class', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Survival Rate (%)')
        axes[0, 1].set_ylim(0, 100)
        for i, v in enumerate(survival_by_class.values):
            axes[0, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # By Embarkation Port
        if 'Embarked' in df.columns:
            survival_by_port = df.groupby('Embarked')['Survived'].mean() * 100
            axes[1, 0].bar(['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)'],
                           survival_by_port.values, color=['#339af0', '#20c997', '#fcc419'])
            axes[1, 0].set_title('Survival Rate by Embarkation Port', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Survival Rate (%)')
            axes[1, 0].set_ylim(0, 100)
            for i, v in enumerate(survival_by_port.values):
                axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # By Title
        if 'Title' in df.columns:
            title_survival = df.groupby('Title')['Survived'].mean() * 100
            title_survival = title_survival.sort_values(ascending=False)
            axes[1, 1].barh(range(len(title_survival)), title_survival.values,
                            color=plt.cm.viridis(np.linspace(0, 1, len(title_survival))))
            axes[1, 1].set_yticks(range(len(title_survival)))
            axes[1, 1].set_yticklabels(title_survival.index)
            axes[1, 1].set_title('Survival Rate by Title', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Survival Rate (%)')
        
        plt.tight_layout()
        self.save_figure(fig, 'survival_by_category.png')
        plt.close()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap of numerical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        self.save_figure(fig, 'correlation_heatmap.png')
        plt.close()
        
        if 'Survived' in correlation_matrix.columns:
            return correlation_matrix['Survived'].sort_values(ascending=False)
    
    def plot_fare_distribution(self, df: pd.DataFrame):
        """Plot fare distribution analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(df['Fare'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_title('Fare Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Fare')
        axes[0].set_ylabel('Frequency')
        
        # Box plot by class
        df.boxplot(column='Fare', by='Pclass', ax=axes[1])
        axes[1].set_title('Fare by Passenger Class', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Passenger Class')
        axes[1].set_ylabel('Fare')
        
        # Fare vs Survival (only if Survived exists)
        if self._has_survived(df):
            try:
                fare_bins = pd.qcut(df['Fare'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
                survival_by_fare = df.groupby(fare_bins)['Survived'].mean() * 100
                axes[2].bar(range(len(survival_by_fare)), survival_by_fare.values,
                           color=plt.cm.Greens(np.linspace(0.3, 0.9, len(survival_by_fare))))
                axes[2].set_xticks(range(len(survival_by_fare)))
                axes[2].set_xticklabels(survival_by_fare.index, rotation=45)
                axes[2].set_title('Survival Rate by Fare Group', fontsize=12, fontweight='bold')
                axes[2].set_ylabel('Survival Rate (%)')
            except Exception as e:
                logger.warning(f"Skipping fare vs survival plot: {e}")
                axes[2].set_visible(False)
        else:
            axes[2].set_visible(False)
        
        plt.tight_layout()
        self.save_figure(fig, 'fare_analysis.png')
        plt.close()
    
    def plot_missing_values(self, df: pd.DataFrame):
        """Plot missing values visualization"""
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_pct = (missing / len(df)) * 100
            bars = ax.barh(missing.index, missing_pct.values, color='coral')
            ax.set_xlabel('Missing Values (%)')
            ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
            for bar, pct in zip(bars, missing_pct.values):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{pct:.1f}%', va='center')
            self.save_figure(fig, 'missing_values.png')
            plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Plot confusion matrix for a model"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Survived', 'Survived'],
                   yticklabels=['Not Survived', 'Survived'], ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}',
               transform=ax.transAxes, ha='center', fontsize=12)
        self.save_figure(fig, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
    
    def plot_roc_curve(self, y_true, y_proba, model_name: str):
        """Plot ROC curve for a model"""
        from sklearn.metrics import roc_curve, roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        self.save_figure(fig, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, top_n: int = 15):
        """Plot feature importance from Random Forest"""
        top_features = feature_importance.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                      color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance (Random Forest)',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        for bar, val in zip(bars, top_features['importance'].values):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        self.save_figure(fig, 'feature_importance.png')
        plt.close()
    
    def create_eda_report(self, df: pd.DataFrame):
        """Create all EDA visualizations"""
        logger.info("Creating EDA visualizations...")
        self.plot_survival_distribution(df)
        self.plot_age_distribution(df)
        self.plot_survival_by_category(df)
        self.plot_correlation_heatmap(df)
        self.plot_fare_distribution(df)
        self.plot_missing_values(df)
        logger.info(f"✅ All EDA visualizations saved to {self.output_dir}")


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    visualizer = Visualizer()
    print("\n🧪 Testing Visualizer...")
    print("✅ Visualizer class ready to use!")