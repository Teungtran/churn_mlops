# webhook_receiver.py
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from src.Churn.utils.logging import logger

import json

router = APIRouter()

@router.post("/webhook/notify", status_code=200)
async def receive_webhook(request: Request):
    try:
        headers = dict(request.headers)
        logger.info(f"Webhook request headers: {headers}")

        body = await request.body()
        body_str = body.decode('utf-8')
        logger.info(f"Webhook request body (raw): {body_str}")

        payload = json.loads(body_str)
        logger.info(f"Parsed JSON payload: {payload}")

        return JSONResponse(content=payload, status_code=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        return JSONResponse(
            content={"error": "Invalid payload or internal error."},
            status_code=status.HTTP_400_BAD_REQUEST
        )
