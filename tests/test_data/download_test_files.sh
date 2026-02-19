#!/usr/bin/env bash

curl https://www.nndc.bnl.gov/endf-b7.1/zips/ENDF-B-VII.1-decay.zip --output ENDFB71.decay.zip
unzip -j ENDFB71.decay.zip decay/dec-053_I_135.endf decay/dec-054_Xe_135.endf
rm -fr ENDF*
