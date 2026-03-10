#####################
# DEPRECATION WARNING
# This file is deprecated and will be removed in a future release.
#####################

from urllib.request import urlretrieve

# Windows: wget must be downloaded and copied to C:/Windows/System32
url = ("https://eternallybored.org/misc/wget/1.21.4/64/wget.exe")
filename = "C:/Windows/System32/wget.exe"
urlretrieve(url, filename)