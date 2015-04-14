"""
Django settings for aptitude_analytics project.

"""

import os
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

ALLOWED_HOSTS = []


INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_extensions',
    'main',
)

MIDDLEWARE_CLASSES = (
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.auth.middleware.SessionAuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
)

ROOT_URLCONF = 'personal_ws.urls'

WSGI_APPLICATION = 'personal_ws.wsgi.application'


###---< Database >---###

DATABASES = {}

###---< Internationalization >---###

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


###---< Static files >---###

MAIN_DIR = os.path.join(PROJECT_DIR, 'main')

STATIC_URL = '/static/'

STATIC_ROOT = os.path.join(MAIN_DIR, 'root')
MEDIA_ROOT = MAIN_DIR + '/media/'

STATICFILES_DIRS = (
    ('main', os.path.join(MAIN_DIR, 'static')),
)

TEMPLATE_DIRS = (
    os.path.join(MAIN_DIR, 'static/templates'),
)

###---< Import Local Settings >---###
try:
    from local_settings import *
except ImportError:
    raise 'Unable to import local settings file'
