from django.shortcuts import render
from pathlib import Path
from .forms import VideoUploadForm  
from .inference import predict_video

UPLOAD_DIR = Path("uploads")

# Create your views here.
def home(request):
    if request.method == "POST":
        form = VideoUploadForm(request.POST,request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES["video"]
            save_path = UPLOAD_DIR / uploaded_file.name
            
            with open(save_path, "wb") as desination:
                for chunk in uploaded_file.chunks():
                    desination.write(chunk)
            
            result = predict_video(video=save_path)
            return render(
                request,
                "detector/result.html",
                {
                    "filename": uploaded_file.name,
                    "prediction": result["prediction"],
                    "probability": result["probability"]
                }
            ) 
        
    else:
        form = VideoUploadForm()
    return render(request,
                  "detector/index.html",
                  {"form": form}
                  )