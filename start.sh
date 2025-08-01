#!/bin/bash

# Navigate to the Django project folder
cd yeastweb

# Start the app using Gunicorn for production
# <module> is the name of the folder that contains wsgi.py
gunicorn --bind=0.0.0.0 --timeout 600 yeastweb.wsgi
