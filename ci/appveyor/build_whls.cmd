cd catboost\python-package
C:\Python27-x64\python.exe mk_wheel.py -DCUDA_ROOT="C:\\CUDA\\v8.0"
C:\Python34-x64\python.exe mk_wheel.py -DCUDA_ROOT="C:\\CUDA\\v8.0"
C:\Python35-x64\python.exe mk_wheel.py -DCUDA_ROOT="C:\\CUDA\\v8.0"
C:\Python36-x64\python.exe mk_wheel.py -DCUDA_ROOT="C:\\CUDA\\v8.0"
cd ..\..
dir catboost\python-package
python ci\webdav_upload.py catboost\python-package\*.whl

