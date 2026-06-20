from django.urls import path
from .views import home, history
from .api_views import predict_api

urlpatterns = [
    path("", home, name="home"),
    path("history/", history, name="history"),
    path("api/predict/", predict_api, name="predict_api"),
]

