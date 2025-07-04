from fastapi import APIRouter, Request, HTTPException
import json
from datetime import datetime

router = APIRouter(
    prefix="/webhook",
    tags=["Webhook Receiver"],
    responses={404: {"description": "Not found"}},
)

# Global webhook data storage
webhook_data = []

@router.post("/notify")
async def receive_webhook(request: Request):
    """
    Receive webhook notifications and store them.
    """
    try:
        body = await request.body()
        payload = json.loads(body.decode('utf-8'))
        
        # Add timestamp if not present
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Store the webhook data
        webhook_data.append(payload)
        
        print(f"Received webhook data: {payload}")
        return payload
    
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")

@router.get("")
async def get_webhooks():
    """
    Get all stored webhook data.
    """
    return webhook_data

@router.delete("")
async def clear_webhooks():
    """
    Clear all stored webhook data.
    """
    webhook_data.clear()
    return {"status": "success", "message": "All webhook data cleared"} 