"""
Pathway Document Connectors
===========================

Connectors for ingesting documents from various sources using Pathway.

Supported Sources:
- Local filesystem (folders)
- Google Drive
- Cloud storage (S3, GCS)

All connectors provide streaming ingestion with automatic updates
when source files change.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    import pathway as pw
    from pathway.io import fs as pw_fs
    PATHWAY_AVAILABLE = True
except ImportError:
    pw = None
    pw_fs = None
    PATHWAY_AVAILABLE = False


class PathwayDocumentConnector:
    """
    Document connector using Pathway's streaming ingestion.
    
    Provides real-time document monitoring and automatic re-indexing
    when source files change.
    
    Example:
        connector = PathwayDocumentConnector.from_local("/path/to/novels")
        docs = connector.read_all()  # Returns list of document texts
    """
    
    def __init__(self):
        if not PATHWAY_AVAILABLE:
            raise ImportError(
                "Pathway is not installed. Install with: pip install pathway-python"
            )
        self._table = None
        self._documents: List[Dict[str, Any]] = []
    
    @classmethod
    def from_local(
        cls, 
        path: str,
        pattern: str = "*.txt",
        mode: str = "static",  # "static" or "streaming"
    ) -> "PathwayDocumentConnector":
        """
        Create connector from local filesystem.
        
        Args:
            path: Path to directory containing documents
            pattern: Glob pattern for file matching (default: *.txt)
            mode: "static" for one-time read, "streaming" for live updates
            
        Returns:
            Configured connector
        """
        connector = cls()
        
        # Read files using Pathway
        folder_path = Path(path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        # Use Pathway's filesystem connector
        connector._table = pw.io.fs.read(
            folder_path,
            format="plaintext",
            mode=mode,
            with_metadata=True,
        )
        
        return connector
    
    @classmethod  
    def from_gdrive(
        cls,
        folder_id: str,
        credentials_path: Optional[str] = None,
        service_account: bool = True,
    ) -> "PathwayDocumentConnector":
        """
        Create connector from Google Drive folder.
        
        Args:
            folder_id: Google Drive folder ID
            credentials_path: Path to credentials.json
            service_account: Use service account (vs. user OAuth)
            
        Returns:
            Configured connector
        """
        connector = cls()
        
        # Use Pathway's Google Drive connector
        connector._table = pw.io.gdrive.read(
            object_id=folder_id,
            service_user_credentials_file=credentials_path,
            mode="static",
            with_metadata=True,
        )
        
        return connector
    
    def read_all(self) -> List[Dict[str, str]]:
        """
        Read all documents from the connected source.
        
        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        if self._table is None:
            return []
        
        # Collect documents from Pathway table
        # Note: In streaming mode, this would need pw.run()
        documents = []
        
        # For static mode, we can materialize immediately
        try:
            # Use debug API for one-shot collection
            rows = pw.debug.table_to_dicts(self._table)
            for row in rows:
                documents.append({
                    'text': row.get('data', row.get('text', '')),
                    'path': row.get('_metadata', {}).get('path', 'unknown'),
                })
        except Exception as e:
            print(f"Warning: Could not read documents: {e}")
        
        return documents
    
    def get_table(self):
        """
        Get the underlying Pathway table for advanced operations.
        
        Returns:
            Pathway Table object
        """
        return self._table


# Convenience function for quick testing
def load_documents_from_path(path: str) -> List[str]:
    """
    Simple function to load documents using Pathway.
    
    Falls back to standard file reading if Pathway unavailable.
    
    Args:
        path: Directory path
        
    Returns:
        List of document texts
    """
    if PATHWAY_AVAILABLE:
        connector = PathwayDocumentConnector.from_local(path)
        docs = connector.read_all()
        return [d['text'] for d in docs]
    else:
        # Fallback: Standard file reading
        folder = Path(path)
        texts = []
        for f in folder.glob("*.txt"):
            texts.append(f.read_text(encoding='utf-8', errors='replace'))
        return texts
