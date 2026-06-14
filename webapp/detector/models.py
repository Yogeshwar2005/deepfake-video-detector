from django.db import models

# Create your models here.

class PredictionHistory(models.Model):
    filename = models.CharField(max_length=255)
    prediction = models.CharField(max_length=50)
    probability = models.FloatField(null=True, blank=True)
    created_at = models.DateField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.filename} - {self.prediction}"
