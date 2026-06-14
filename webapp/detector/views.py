from django.shortcuts import render
from pathlib import Path
from uuid import uuid4
from .forms import VideoUploadForm  
from .inference import predict_video
from .models import PredictionHistory

UPLOAD_DIR = Path("uploads")

# Create your views here.
def home(request):
    if request.method == "POST":
        form = VideoUploadForm(request.POST,request.FILES)
        
        valid_formats = [".mp4", ".mov", ".avi", ".mkv"]
      
        if form.is_valid():
            uploaded_file = request.FILES["video"]
            
            file_extension = Path(uploaded_file.name).suffix.lower()
            if file_extension not in valid_formats:
                return render(
                            request,
                            "detector/index.html",
                            {
                              "form": form,
                              "error": "Invalid file format. Please upload mp4, mov, avi, or mkv."
                            }
                        )
            
            save_path = UPLOAD_DIR / f"{uuid4()}{file_extension}"
            
            try:
                with open(save_path, "wb") as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                
                result = predict_video(video=save_path)
                probability = round(result["probability"]*100, 2)
                
                PredictionHistory.objects.create(
                    filename = uploaded_file.name,
                    prediction = result["prediction"],
                    probability = probability
                    )
            
            finally:
                if save_path.exists():
                    save_path.unlink()
            
            return render(
                request,
                "detector/result.html",
                {
                    "filename": uploaded_file.name,
                    "prediction": result["prediction"],
                    "probability": probability
                }
            ) 
        
    else:
        form = VideoUploadForm()
    return render(
            request,
            "detector/index.html",
            {
                "form": form
            }
        )
    

def history(request):
    records = PredictionHistory.objects.all().order_by("-created_at")
    
    return render(
        request,
        "detector/history.html",
        {
            "records": records
        }
    )