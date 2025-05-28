from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class LogCategory(Enum):
    INPUT = "input"
    PROCESS = "process"
    RESULT = "result"
    SCORE = "score"
    SYSTEM = "system"

@dataclass
class LogEntry:
    """Represents a single log entry"""
    id: str
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    content: Dict[str, Any]
    context_id: Optional[str] = None
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MemoryLogger:
    """Logger for memory system events and processes"""
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("memory_logs")
        self.logs: List[LogEntry] = []
        self.context_stack: List[str] = []
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure log directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def log_input(self,
                  input_data: Any,
                  context_id: Optional[str] = None,
                  tags: Optional[List[str]] = None) -> LogEntry:
        """Log input data"""
        return self._create_log_entry(
            level=LogLevel.INFO,
            category=LogCategory.INPUT,
            content={'input': input_data},
            context_id=context_id,
            tags=tags or []
        )
        
    def log_process(self,
                   process_name: str,
                   details: Dict[str, Any],
                   parent_id: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> LogEntry:
        """Log process details"""
        return self._create_log_entry(
            level=LogLevel.INFO,
            category=LogCategory.PROCESS,
            content={
                'process_name': process_name,
                'details': details
            },
            parent_id=parent_id,
            tags=tags or []
        )
        
    def log_result(self,
                  result: Any,
                  process_id: str,
                  success: bool = True,
                  tags: Optional[List[str]] = None) -> LogEntry:
        """Log process result"""
        return self._create_log_entry(
            level=LogLevel.INFO if success else LogLevel.WARNING,
            category=LogCategory.RESULT,
            content={
                'result': result,
                'success': success
            },
            parent_id=process_id,
            tags=tags or []
        )
        
    def log_score(self,
                 target_id: str,
                 scores: Dict[str, float],
                 details: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None) -> LogEntry:
        """Log evaluation scores"""
        return self._create_log_entry(
            level=LogLevel.INFO,
            category=LogCategory.SCORE,
            content={
                'target_id': target_id,
                'scores': scores,
                'details': details or {}
            },
            tags=tags or []
        )
        
    def start_context(self, context_id: str):
        """Start a new logging context"""
        self.context_stack.append(context_id)
        
    def end_context(self) -> Optional[str]:
        """End the current logging context"""
        if self.context_stack:
            return self.context_stack.pop()
        return None
        
    def get_current_context(self) -> Optional[str]:
        """Get current context ID"""
        return self.context_stack[-1] if self.context_stack else None
        
    def _create_log_entry(self,
                         level: LogLevel,
                         category: LogCategory,
                         content: Dict[str, Any],
                         context_id: Optional[str] = None,
                         parent_id: Optional[str] = None,
                         tags: Optional[List[str]] = None) -> LogEntry:
        """Create and store log entry"""
        # Generate log ID
        log_id = self._generate_log_id()
        
        # Create entry
        entry = LogEntry(
            id=log_id,
            timestamp=datetime.now(),
            level=level,
            category=category,
            content=content,
            context_id=context_id or self.get_current_context(),
            parent_id=parent_id,
            tags=tags or []
        )
        
        # Add system metadata
        entry.metadata.update({
            'session_id': self.current_session,
            'sequence_number': len(self.logs)
        })
        
        # Store entry
        self.logs.append(entry)
        
        # Save to storage
        self._save_log_entry(entry)
        
        return entry
        
    def _generate_log_id(self) -> str:
        """Generate unique log ID"""
        import hashlib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"log_{timestamp}"
        
    def _save_log_entry(self, entry: LogEntry):
        """Save log entry to storage"""
        try:
            # Create log file path
            log_file = self.storage_path / f"{self.current_session}.jsonl"
            
            # Convert entry to JSON
            log_data = {
                'id': entry.id,
                'timestamp': entry.timestamp.isoformat(),
                'level': entry.level.value,
                'category': entry.category.value,
                'content': entry.content,
                'context_id': entry.context_id,
                'parent_id': entry.parent_id,
                'tags': entry.tags,
                'metadata': entry.metadata
            }
            
            # Append to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save log entry: {str(e)}")
            
    def get_logs(self,
                 context_id: Optional[str] = None,
                 category: Optional[LogCategory] = None,
                 level: Optional[LogLevel] = None,
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None) -> List[LogEntry]:
        """Query logs with filters"""
        filtered_logs = self.logs
        
        if context_id:
            filtered_logs = [
                log for log in filtered_logs
                if log.context_id == context_id
            ]
            
        if category:
            filtered_logs = [
                log for log in filtered_logs
                if log.category == category
            ]
            
        if level:
            filtered_logs = [
                log for log in filtered_logs
                if log.level == level
            ]
            
        if start_time:
            filtered_logs = [
                log for log in filtered_logs
                if log.timestamp >= start_time
            ]
            
        if end_time:
            filtered_logs = [
                log for log in filtered_logs
                if log.timestamp <= end_time
            ]
            
        return filtered_logs
        
    def get_process_chain(self, process_id: str) -> List[LogEntry]:
        """Get chain of logs for process"""
        chain = []
        current_id = process_id
        
        while current_id:
            # Find log entry
            entry = next(
                (log for log in self.logs if log.id == current_id),
                None
            )
            if not entry:
                break
                
            chain.append(entry)
            current_id = entry.parent_id
            
        return list(reversed(chain))  # Return in chronological order
        
    def get_context_summary(self, context_id: str) -> Dict[str, Any]:
        """Get summary of logs in context"""
        context_logs = self.get_logs(context_id=context_id)
        
        if not context_logs:
            return {}
            
        return {
            'context_id': context_id,
            'start_time': min(log.timestamp for log in context_logs),
            'end_time': max(log.timestamp for log in context_logs),
            'total_logs': len(context_logs),
            'categories': {
                category.value: len([
                    log for log in context_logs
                    if log.category == category
                ])
                for category in LogCategory
            },
            'levels': {
                level.value: len([
                    log for log in context_logs
                    if log.level == level
                ])
                for level in LogLevel
            }
        }
        
    def clear_logs(self):
        """Clear all logs"""
        self.logs = []
        self.context_stack = [] 