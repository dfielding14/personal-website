SECRET_KEY = '+-nn!703*mdznj)o*doue7d9)^nmu1367bl3yc3^y&&esrowla'

DEBUG = True
TEMPLATE_DEBUG = True

DATABASES = {
        'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'personal_ws',
        'USER': 'lb',
        'PASSWORD': 'password',
        'HOST': '127.0.0.1',
        'PORT': '5432',
        }
}