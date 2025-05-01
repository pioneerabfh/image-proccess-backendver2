# api/views.py
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from PIL import Image
import io
import base64

@csrf_exempt
def image_proccess(request):
    if request.method == 'POST':
        try:
            
            import json
            data = json.loads(request.body)

           
            image_data = data.get("imageBase64")
            width = int(data.get("width"))
            height = int(data.get("height"))
            startX = int(data.get("startX"))
            startY = int(data.get("startY"))

            
            format, imgstr = image_data.split(';base64,') 
            image_bytes = io.BytesIO(base64.b64decode(imgstr))
            img = Image.open(image_bytes)

            
            cropped = img.crop((startX, startY, startX + width, startY + height))

            
            buffer = io.BytesIO()
            cropped.save(buffer, format="PNG")
            encoded_cropped = base64.b64encode(buffer.getvalue()).decode('utf-8')
            result_image = "data:image/png;base64," + encoded_cropped

            return JsonResponse({"croppedImage": result_image})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Only POST allowed"}, status=405)
def home(request):
    return JsonResponse({"message": "Backend Cevabı Döndü"})