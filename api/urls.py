from django.urls import path
from . import views

urlpatterns = [
    path('proccess/', views.image_proccess, name='image_proccess'),
    path('home/',views.home,name='home')
    
]

