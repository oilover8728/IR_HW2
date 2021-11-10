from django.urls import path

from . import views

urlpatterns = [
    path('home', views.home, name='home'),
    path('upload_file', views.upload_file, name='upload_file'),
    path('search', views.search, name='search'),
    path('clear', views.clear, name='clear'),
    path('index', views.index, name='index'),
    path('insert', views.insert, name='insert'),
    path('browser_search', views.browser_search, name='browser_search'),
    path('inverted_constrcut', views.inverted_constrcut, name='inverted_constrcut'),
    path('check_table', views.check_table, name='check_table'),
    path('graph', views.graph, name='graph'),
]