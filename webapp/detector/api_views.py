from pathlib import Path
from uuid import uuid4

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .inference import predict_video

UPLOAD_DIR = Path("media/uploads")

@api_view(["POST"])
def predict_api(request):
    if "video" not in request.FILES:
        return Response(
            {
                "error": "No video file provided"
            },
            status=status.HTTP_400_BAD_REQUEST
        )
    
    uploaded_file = request.FILES["video"]
    
    valid_formats = [".mp4", ".mov", ".avi", ".mkv"]
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension not in valid_formats:
        return Response(
            {
                "error": "Invalid file format"
            },
            status=status.HTTP_400_BAD_REQUEST
        )
    
    result_id = str(uuid4())
    save_path = UPLOAD_DIR / f"{result_id}{file_extension}"
    
    try:
        with open(save_path, "wb") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        result = predict_video(video=save_path, result_id=result_id)

        return Response({
            "filename": uploaded_file.name,
            "prediction": result["prediction"],
            "fake_probability": round(result["probability"] * 100, 2),
            "frames_analyzed": result["frames_analyzed"],
            "faces_detected": result["faces_detected"],
            "threshold": round(result["threshold"] * 100, 2),
            "gradcam_results": result.get("gradcam_results", []),
        })
    except Exception as e:
        return Response(
            {
                "error": str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        if save_path.exists():
            save_path.unlink()