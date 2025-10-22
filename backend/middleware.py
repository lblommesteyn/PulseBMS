import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("middleware")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            
            # Log response
            process_time = (time.time() - start_time) * 1000
            logger.info(
                f"Response: {response.status_code} "
                f"Time: {process_time:.2f}ms"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error", "error": str(e)}
            )

class CORSPreflightMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            response = await call_next(request)
            return response
        return await call_next(request)
