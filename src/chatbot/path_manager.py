"""
Path Manager for Punch IQ AI
Centralizes all file path management and validation
"""
import os
import streamlit as st
from pathlib import Path
from typing import Dict, Optional, List


class PathManager:
    """Centralized path management for the chatbot"""
    
    def __init__(self):
        # Determine project root from script location
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent
        self.data_dir = self.project_root / "data"
        self.model_dir = self.project_root / "model"
        self.secret_dir = self.project_root / "secret"
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [self.data_dir, self.model_dir, self.secret_dir]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def paths(self) -> Dict[str, Path]:
        """Return all important paths"""
        return {
            'project_root': self.project_root,
            'script_dir': self.script_dir,
            'data_dir': self.data_dir,
            'model_dir': self.model_dir,
            'secret_dir': self.secret_dir,
            
            # Data files
            'fighters_csv': self.data_dir / "fighters_cleaned.csv",
            'fighters_raw': self.data_dir / "fighters.csv",
            'fighters_embeddings': self.data_dir / "fighters_embeddings.npy",
            'faiss_db': self.data_dir / "faiss_vectorized_db",
            'faiss_index': self.data_dir / "faiss_vectorized_db" / "index.faiss",
            'faiss_pkl': self.data_dir / "faiss_vectorized_db" / "index.pkl",
            
            # Model files
            'boxing_model': self.model_dir / "ML" / "boxing_model.pkl",
            'label_encoder': self.model_dir / "ML" / "label_encoder.pkl",
            'scaler': self.model_dir / "ML" / "scaler.pkl",
            
            # Secret files
            'mistral_key': self.secret_dir / "MISTRAL_API_KEY",
            'openai_key': self.secret_dir / "OPEN_AI_API_KEY"
        }
    
    def get_path(self, key: str) -> Path:
        """Get a specific path by key"""
        return self.paths.get(key)
    
    def get_absolute_path(self, key: str) -> str:
        """Get absolute path as string"""
        path = self.get_path(key)
        return str(path.absolute()) if path else None
    
    def validate_file(self, key: str) -> Dict[str, any]:
        """Validate if a file exists and get its info"""
        path = self.get_path(key)
        if not path:
            return {'exists': False, 'error': f'Unknown path key: {key}'}
        
        absolute_path = path.absolute()
        exists = absolute_path.exists()
        
        result = {
            'exists': exists,
            'path': str(absolute_path),
            'relative_path': str(path),
            'key': key
        }
        
        if exists:
            if absolute_path.is_file():
                result['size'] = absolute_path.stat().st_size
                result['type'] = 'file'
            else:
                result['type'] = 'directory'
        else:
            result['error'] = f'File not found: {absolute_path}'
        
        return result
    
    def find_csv_files(self) -> List[Dict[str, str]]:
        """Find all CSV files in the data directory"""
        csv_files = []
        if self.data_dir.exists():
            for file_path in self.data_dir.glob("*.csv"):
                csv_files.append({
                    'name': file_path.name,
                    'path': str(file_path.absolute()),
                    'size': file_path.stat().st_size
                })
        return csv_files
    
    def get_default_dataset(self) -> Optional[str]:
        """Get the default dataset path, with fallback options"""
        # Priority order for dataset selection
        candidates = ['fighters_csv', 'fighters_raw']
        
        for candidate in candidates:
            validation = self.validate_file(candidate)
            if validation['exists']:
                return validation['path']
        
        # If no predefined files exist, look for any CSV
        csv_files = self.find_csv_files()
        if csv_files:
            return csv_files[0]['path']
        
        return None
    
    def create_diagnostic_info(self) -> Dict[str, any]:
        """Create comprehensive diagnostic information"""
        info = {
            'project_root': str(self.project_root.absolute()),
            'current_working_dir': os.getcwd(),
            'script_location': str(self.script_dir.absolute()),
            'directories': {},
            'files': {},
            'csv_files': self.find_csv_files()
        }
        
        # Check directories
        for key in ['data_dir', 'model_dir', 'secret_dir']:
            path = self.get_path(key)
            info['directories'][key] = {
                'path': str(path.absolute()),
                'exists': path.exists(),
                'is_writable': os.access(path, os.W_OK) if path.exists() else False
            }
        
        # Check important files
        file_keys = ['fighters_csv', 'fighters_raw', 'fighters_embeddings', 'faiss_db']
        for key in file_keys:
            info['files'][key] = self.validate_file(key)
        
        return info
    
    def display_diagnostic(self):
        """Display diagnostic information in Streamlit"""
        info = self.create_diagnostic_info()
        
        with st.expander("🔍 Path Diagnostic Information"):
            st.subheader("📁 Directories")
            for key, dir_info in info['directories'].items():
                status = "✅" if dir_info['exists'] else "❌"
                writable = "✍️" if dir_info.get('is_writable') else "🔒"
                st.text(f"{status} {writable} {key}: {dir_info['path']}")
            
            st.subheader("📄 Important Files")
            for key, file_info in info['files'].items():
                status = "✅" if file_info['exists'] else "❌"
                size_info = f" ({file_info.get('size', 0)} bytes)" if file_info['exists'] else ""
                st.text(f"{status} {key}: {file_info['path']}{size_info}")
            
            st.subheader("📊 Available CSV Files")
            if info['csv_files']:
                for csv_file in info['csv_files']:
                    st.text(f"📋 {csv_file['name']}: {csv_file['path']} ({csv_file['size']} bytes)")
            else:
                st.warning("No CSV files found in data directory")
            
            st.subheader("🗂️ System Information")
            st.text(f"📍 Project Root: {info['project_root']}")
            st.text(f"📍 Working Directory: {info['current_working_dir']}")
            st.text(f"📍 Script Location: {info['script_location']}")


# Global instance
path_manager = PathManager()


def get_path_manager() -> PathManager:
    """Get the global path manager instance"""
    return path_manager
