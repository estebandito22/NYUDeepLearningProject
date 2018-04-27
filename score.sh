for i in $(seq 12 4 172); do PYTHONPATH=$PYTHONPATH:. python score.py -m nonspatial_alexnet_mp_noteverystepcsal -e $i; done
