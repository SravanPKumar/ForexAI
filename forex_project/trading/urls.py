from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('trade/', views.trade, name='trade'),
    path('login/', views.login, name='login'),
    path('chart/', views.chart, name='chart'),
]
