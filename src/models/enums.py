from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Provider(str, Enum):
    DALLE = "dalle"
    MIDJOURNEY = "midjourney"
    OPENROUTER = "openrouter" 
