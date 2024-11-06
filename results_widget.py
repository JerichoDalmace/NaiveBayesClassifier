from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                                 QTableWidget, QTableWidgetItem, QPushButton,
                                 QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ResultsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Summary section
        summary_layout = QHBoxLayout()
        self.total_label = QLabel("Total Reviews: 0")
        self.positive_label = QLabel("Positive Reviews: 0")
        self.negative_label = QLabel("Negative Reviews: 0")
        
        for label in [self.total_label, self.positive_label, self.negative_label]:
            label.setStyleSheet("font-size: 16px; margin: 10px;")
            summary_layout.addWidget(label)
        
        layout.addLayout(summary_layout)
        
        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Review', 'Cleaned Review', 'Sentiment', 'Confidence'])
        layout.addWidget(self.table)
        
        # Graph
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Export button
        self.export_btn = QPushButton("Export Results to CSV")
        self.export_btn.clicked.connect(self.export_results)
        layout.addWidget(self.export_btn)
        
        self.df = None
    
    def update_results(self, df):
        self.df = df
        
        # Update summary
        total = len(df)
        positive = sum(df['sentiment'] == 1)
        negative = sum(df['sentiment'] == 0)
        
        self.total_label.setText(f"Total Reviews: {total}")
        self.positive_label.setText(f"Positive Reviews: {positive}")
        self.negative_label.setText(f"Negative Reviews: {negative}")
        
        # Update table
        self.table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            self.table.setItem(i, 0, QTableWidgetItem(str(row['review'])))
            self.table.setItem(i, 1, QTableWidgetItem(str(row['cleaned_review'])))
            sentiment = "Positive" if row['sentiment'] == 1 else "Negative"
            self.table.setItem(i, 2, QTableWidgetItem(sentiment))
            self.table.setItem(i, 3, QTableWidgetItem(f"{row['confidence']:.2f}"))
        
        # Update graph
        self.ax.clear()
        df['confidence'].hist(bins=20, ax=self.ax)
        self.ax.set_xlabel('Confidence Score')
        self.ax.set_ylabel('Number of Reviews')
        self.ax.set_title('Confidence Score Distribution')
        self.canvas.draw()
    
    def export_results(self):
        if self.df is not None:
            try:
                file_name, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Results",
                    "",
                    "CSV Files (*.csv);;All Files (*)"
                )
                if file_name:
                    self.df.to_csv(file_name, index=False)
                    QMessageBox.information(self, "Success", "Results exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")