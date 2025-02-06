#!/bin/bash -eu

git clone https://github.com/alexwarstadt/blimp data/blimp
git clone https://github.com/osekilab/JBLiMP data/jblimp
wget -O data/climp.zip https://github.com/beileixiang/CLiMP/raw/main/CLiMP_corpus.zip \
  && unzip data/climp.zip -d data/climp/ \
  && rm data/climp.zip
