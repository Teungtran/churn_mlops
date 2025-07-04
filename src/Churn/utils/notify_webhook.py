from typing import Dict, Any, Optional, Union
import httpx

from .logging import logger


HEADERS_JSON: Dict[str, str] = {"Content-Type": "application/json"}


async def post_to_webhook(
    webhook_url: str, 
    WEBHOOK_PAYLOAD: Dict, 
    headers: Optional[Dict[str, str]] = None
) -> Union[httpx.Response, None]:
    """
    Posts a payload to a webhook URL.

    Returns:
        The HTTP response object on success.
        None on failure.

    Raises:
        Exception: If an exception occurs during the request.
    """
    headers = headers or HEADERS_JSON

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            logger.info(f"Posting to webhook: {webhook_url} with payload: {WEBHOOK_PAYLOAD}")
            response = await client.post(webhook_url, json=WEBHOOK_PAYLOAD, headers=headers)

            logger.info(f"Webhook response status: {response.status_code}, body: {response.text}")

            if response.status_code in (200, 201):
                return response
            else:
                raise Exception(f"Failed to notify webhook. Status: {response.status_code}, Body: {response.text}")
        
        except Exception as e:
            logger.error(f"Exception posting to webhook: {str(e)}")
            raise


