from django.urls import path
from . import views

urlpatterns = [
    path('process/', views.image_proccess, name='image_proccess'),
    
    
]

