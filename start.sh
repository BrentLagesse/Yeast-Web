#!/bin/bash

# Navigate to the Django project folder
cd yeastweb

# Install only necessary dependencies (remove unnecessary ones)
apt-get update
apt-get install -y libpq-dev  # If using PostgreSQL, for example

# Ensure that static files are collected (optional, but recommended)
python manage.py collectstatic --noinput

# Start the app using Gunicorn for production
gunicorn --workers 3 --bind 0.0.0.0:8000 yeastweb.wsgi:application
