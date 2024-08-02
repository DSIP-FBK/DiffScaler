#!/bin/bash

# download data
wget "https://zenodo.org/records/12944960/files/2000-2002.zip?download=1" -O 2000-2002.zip
wget "https://zenodo.org/records/12945014/files/2003-2005.zip?download=1" -O 2003-2005.zip
wget "https://zenodo.org/records/12945028/files/2006-2008.zip?download=1" -O 2006-2008.zip
wget "https://zenodo.org/records/12945040/files/2009-2011.zip?download=1" -O 2009-2011.zip
wget "https://zenodo.org/records/12945050/files/2012-2014.zip?download=1" -O 2012-2014.zip
wget "https://zenodo.org/records/12945058/files/2015-2017.zip?download=1" -O 2015-2017.zip
wget "https://zenodo.org/records/12945066/files/2018-2020.zip?download=1" -O 2018-2020.zip

# unzip data
unzip 2000-2002.zip -d ./
rm 2000-2002.zip
unzip 2003-2005.zip -d ./
rm 2003-2005.zip
unzip 2006-2008.zip -d ./
rm 2006-2008.zip
unzip 2009-2011.zip -d ./
rm 2009-2011.zip
unzip 2012-2014.zip -d ./
rm 2012-2014.zip
unzip 2015-2017.zip -d ./
rm 2015-2017.zip
unzip 2018-2020.zip -d ./
rm 2018-2020.zip