"""
Simple database logging for no-auth mode.
Logs API requests directly to Supabase without authentication.
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import json

from webscout.Litlogger import Logger, LogLevel, LogFormat, ConsoleHandler
import sys

# Setup logger
logger = Logger(
    name="webscout.api.simple_db",
    level=LogLevel.INFO,
    handlers=[ConsoleHandler(stream=sys.stdout)],
    fmt=LogFormat.DEFAULT
)

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    logger.warning("Supabase not available. Install with: pip install supabase")
    SUPABASE_AVAILABLE = False


class SimpleRequestLogger:
    """Simple request logger for no-auth mode."""
    
    def __init__(self):
        self.supabase_client: Optional[Client] = None
        self.initialize_supabase()
    
    def initialize_supabase(self):
        """Initialize Supabase client - REQUIRED for logging."""
        if not SUPABASE_AVAILABLE:
            logger.error("❌ Supabase package not installed! Install with: pip install supabase")
            logger.error("Request logging is DISABLED without Supabase.")
            return
            
        supabase_url = self._get_supabase_url()
        supabase_key = self._get_supabase_key()
        
        if supabase_url and supabase_key:
            try:
                self.supabase_client = create_client(supabase_url, supabase_key)
                logger.info("✅ Supabase client initialized for request logging")
                logger.info("🎯 All API requests will be logged to Supabase")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase client: {e}")
                logger.error("Request logging is DISABLED due to Supabase connection failure.")
                self.supabase_client = None
        else:
            logger.error("❌ Supabase credentials not found!")
            logger.error("Request logging is DISABLED. Set SUPABASE_URL and SUPABASE_ANON_KEY.")
    
    def _get_supabase_url(self) -> Optional[str]:
        """Get Supabase URL from environment variables or GitHub secrets."""
        # Try environment variable first
        url = os.getenv("SUPABASE_URL")
        if url:
            logger.info("📍 Using SUPABASE_URL from environment")
            return url
        
        # Try to get from GitHub secrets (if running in GitHub Actions)
        github_url = os.getenv("GITHUB_SUPABASE_URL")  # GitHub Actions secret
        if github_url:
            logger.info("📍 Using SUPABASE_URL from GitHub secrets")
            return github_url
        
        logger.error("⚠️  SUPABASE_URL not found in environment or GitHub secrets")
        return None
    
    def _get_supabase_key(self) -> Optional[str]:
        """Get Supabase key from environment variables or GitHub secrets."""
        # Try environment variable first
        key = os.getenv("SUPABASE_ANON_KEY")
        if key:
            logger.info("🔑 Using SUPABASE_ANON_KEY from environment")
            return key
        
        # Try to get from GitHub secrets (if running in GitHub Actions)
        github_key = os.getenv("GITHUB_SUPABASE_ANON_KEY")  # GitHub Actions secret
        if github_key:
            logger.info("🔑 Using SUPABASE_ANON_KEY from GitHub secrets")
            return github_key
        
        logger.error("⚠️  SUPABASE_ANON_KEY not found in environment or GitHub secrets")
        return None
    
    async def log_request(
        self,
        request_id: str,
        ip_address: str,
        model: str,
        question: str,
        answer: str,
        provider: Optional[str] = None,
        request_time: Optional[datetime] = None,
        response_time: Optional[datetime] = None,
        processing_time_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
        error: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """
        Log API request details to Supabase.
        
        Args:
            request_id: Unique identifier for the request
            ip_address: Client IP address
            model: Model used for the request
            question: User's question/prompt
            answer: AI's response
            provider: Provider used (e.g., ChatGPT, Claude, etc.)
            request_time: When the request was received
            response_time: When the response was sent
            processing_time_ms: Processing time in milliseconds
            tokens_used: Number of tokens consumed
            error: Error message if any
            user_agent: User agent string
            
        Returns:
            bool: True if logged successfully, False otherwise
        """
        if not self.supabase_client:
            # Still log to console for debugging
            logger.info(f"Request {request_id}: {model} - {question[:100]}...")
            return False
        
        if not request_time:
            request_time = datetime.now(timezone.utc)
        
        if not response_time:
            response_time = datetime.now(timezone.utc)
        
        try:
            data = {
                "request_id": request_id,
                "ip_address": ip_address,
                "model": model,
                "provider": provider or "unknown",
                "question": question[:2000] if question else "",  # Truncate long questions
                "answer": answer[:5000] if answer else "",  # Truncate long answers
                "request_time": request_time.isoformat(),
                "response_time": response_time.isoformat(),
                "processing_time_ms": processing_time_ms,
                "tokens_used": tokens_used,
                "error": error[:1000] if error else None,  # Truncate long errors
                "user_agent": user_agent[:500] if user_agent else None,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            result = self.supabase_client.table("api_requests").insert(data).execute()
            
            if result.data:
                logger.info(f"✅ Request {request_id} logged to database")
                return True
            else:
                logger.error(f"❌ Failed to log request {request_id}: No data returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to log request {request_id}: {e}")
            return False
    
    async def get_recent_requests(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent API requests for monitoring."""
        if not self.supabase_client:
            return {"error": "Database not available", "requests": []}
        
        try:
            result = self.supabase_client.table("api_requests")\
                .select("request_id, ip_address, model, provider, created_at, processing_time_ms, error")\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            return {
                "requests": result.data if result.data else [],
                "count": len(result.data) if result.data else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get recent requests: {e}")
            return {"error": str(e), "requests": []}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about API usage."""
        if not self.supabase_client:
            return {"error": "Database not available"}
        
        try:
            # Get total requests today
            today = datetime.now(timezone.utc).date().isoformat()
            
            today_requests = self.supabase_client.table("api_requests")\
                .select("request_id", count="exact")\
                .gte("created_at", f"{today}T00:00:00Z")\
                .execute()
            
            # Get requests by model (last 100)
            model_requests = self.supabase_client.table("api_requests")\
                .select("model")\
                .order("created_at", desc=True)\
                .limit(100)\
                .execute()
            
            model_counts = {}
            if model_requests.data:
                for req in model_requests.data:
                    model = req.get("model", "unknown")
                    model_counts[model] = model_counts.get(model, 0) + 1
            
            return {
                "today_requests": today_requests.count if hasattr(today_requests, 'count') else 0,
                "model_usage": model_counts,
                "available": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e), "available": False}


# Global instance
request_logger = SimpleRequestLogger()


async def log_api_request(
    request_id: str,
    ip_address: str,
    model: str,
    question: str,
    answer: str,
    **kwargs
) -> bool:
    """Convenience function to log API requests."""
    return await request_logger.log_request(
        request_id=request_id,
        ip_address=ip_address,
        model=model,
        question=question,
        answer=answer,
        **kwargs
    )


def get_client_ip(request) -> str:
    """Extract client IP address from request."""
    # Check for X-Forwarded-For header (common with proxies/load balancers)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    # Check for X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fall back to direct client IP
    return getattr(request.client, "host", "unknown")


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())
