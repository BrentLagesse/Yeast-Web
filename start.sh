#!/bin/bash
cd yeastweb
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0
python manage.py runserver
