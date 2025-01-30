# chatbot/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # path('', views.index, name='index'),  # Home page view
    path('chatbot_response/', views.chatbot_response, name='chatbot_response'),  # API to get response from chatbot
]
