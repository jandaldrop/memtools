rm ./ckernel.cpython*
rm ./build/*
rm ./__pycache__/*
python3 setup-ckernel.py build_ext --inplace
