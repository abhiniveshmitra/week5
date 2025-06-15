import requests
from dotenv import load_dotenv
import os

load_dotenv()
VISION_KEY = os.getenv("AZURE_VISION_KEY")
VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")

def azure_ocr_image(image_path):
    ocr_url = VISION_ENDPOINT.rstrip('/') + "/vision/v3.2/read/analyze"
    headers = {
        'Ocp-Apim-Subscription-Key': VISION_KEY,
        'Content-Type': 'application/octet-stream'
    }
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    response = requests.post(ocr_url, headers=headers, data=img_data)
    if response.status_code != 202:
        return "OCR request failed: " + response.text
    operation_url = response.headers["Operation-Location"]

    # Poll for result (usually takes 1-3s)
    import time
    for _ in range(10):
        result = requests.get(operation_url, headers={'Ocp-Apim-Subscription-Key': VISION_KEY})
        res_json = result.json()
        if res_json.get("status") == "succeeded":
            lines = []
            for read_result in res_json["analyzeResult"]["readResults"]:
                for l in read_result["lines"]:
                    lines.append(l["text"])
            return "\n".join(lines)
        time.sleep(1)
    return "Timed out waiting for OCR result."
