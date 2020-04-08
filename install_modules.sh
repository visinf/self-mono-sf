#!/bin/bash
cd ./models/correlation_package
python setup.py install
cd ../forwardwarp_package
python setup.py install
cd ../..
