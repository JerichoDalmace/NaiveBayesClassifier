from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                                 QLabel, QFileDialog, QProgressBar, QTableWidget, 
                                 QTableWidgetItem, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pandas as pd
import sys
from analysis_thread import AnalysisThread
from results_widget import ResultsWidget
from model import SentimentClassifier
from data_preprocessing import TextPreprocessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Student Review Sentiment Analyzer")
        self.setMinimumSize(800, 600)
        
        # Initialize components
        self.init_ui()
        self.load_model()
        
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add title label
        title = QLabel("Student Review Sentiment Analyzer")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Add upload button
        self.upload_btn = QPushButton("Upload CSV File")
        self.upload_btn.setStyleSheet("padding: 10px; font-size: 16px;")
        self.upload_btn.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_btn)
        
        # Add progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Add results widget (hidden by default)
        self.results_widget = ResultsWidget()
        self.results_widget.hide()
        layout.addWidget(self.results_widget)
        
    def load_model(self):
        try:
            self.model = SentimentClassifier.load_model('sentiment_classifier.pkl')
            self.preprocessor = TextPreprocessor()
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               "Failed to load model. Please ensure the model is trained.\n"
                               f"Error: {str(e)}")
            sys.exit(1)
    
    def upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_name:
            try:
                # Create and start analysis thread
                self.analysis_thread = AnalysisThread(
                    file_name, self.model, self.preprocessor
                )
                self.analysis_thread.progress_updated.connect(self.update_progress)
                self.analysis_thread.analysis_complete.connect(self.show_results)
                self.analysis_thread.error_occurred.connect(self.show_error)
                
                # Show progress bar and start analysis
                self.progress_bar.show()
                self.progress_bar.setValue(0)
                self.upload_btn.setEnabled(False)
                self.analysis_thread.start()
                
            except Exception as e:
                self.show_error(str(e))
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_results(self, results_df):
        self.progress_bar.hide()
        self.upload_btn.setEnabled(True)
        self.results_widget.show()
        self.results_widget.update_results(results_df)
    
    def show_error(self, error_message):
        self.progress_bar.hide()
        self.upload_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", str(error_message))